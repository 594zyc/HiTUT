import os, sys, json, pickle, io, time, random, copy
import h5py
import pprint
import threading, queue
from tqdm import tqdm
from collections import Counter
from transformers import MobileBertTokenizer

import cv2
from PIL import Image
import numpy as np
import revtok
import torch

sys.path.append(os.path.join(os.environ['ALFRED_ROOT']))

from gen.constants import *
from gen.utils.py_util import remove_spaces_and_lower
from gen.utils.bb_util import bb_IoU

from models.config.configs import Config
from models.utils.vocab import Vocab
from models.nn.mrcnn import MaskRCNNDetector
from models.utils.bert_utils import get_bert_tknz, mmt_word_ids_to_bert_ids

lock = threading.Lock()

class AlfredPyTorchDataset(torch.utils.data.Dataset):
    def __init__(self, alfred_data, split, task_type, args):
        self.data = alfred_data
        self.split = split
        self.task_type = task_type

        self.args = args
        self.detector_type = args.detector_type
        self.topk = args.topk_objs
        self.max_length = args.max_enc_length
        self.image_size = args.image_size
        self.action_vocab = self.data.dec_in_vocab

        self.level = task_type.split('_')[0]
        self.low_data = task_type.split('_')[1] if self.level == 'low' else None

        self.dataset = self.get_data_instances()

        # if self.args.use_bert:
        #     with open('data/full_2.1.0_pp/language_%s.json'%self.split, 'r') as f:
        #         self.language_data = json.load(f)

        # self.resnet18_feats = h5py.File('data/resnet18_feats.hdf5', 'r')
        # with open('data/full_2.1.0_pp/img_map_%s.json'%split, 'r') as f:
        #     self.img_map = json.load(f)


    def get_data_instances(self):
        det_sp = os.path.join(self.data.pp_path, '%s_det_res_%s.json'%(self.split, self.args.detector_type))
        with open(det_sp, 'rb') as f:
            self.obj_det_res = json.load(f)

        if self.level == 'high':
            sp = os.path.join(self.data.pp_path, '%s_high_action_instances.json'%(self.split))
        elif self.level == 'low':
            sp = os.path.join(self.data.pp_path, '%s_low_action_instances_%s.json'%(self.split, self.low_data))
            if self.args.train_one_shot and self.split == 'train':
                sp = sp.replace('instances', 'seed')

        if not os.path.exists(sp) or not os.path.exists(det_sp):
            self.data.prepare_data_instances()
        with open(sp, 'rb') as f:
            self.dataset = json.load(f)

        if self.split == 'train':
            if self.args.train_one_shot:
                total_len = 209331 if self.low_data == 'mani' else 983260
                self.args.train_proportion = len(self.dataset) / total_len * 100
            elif self.args.train_proportion != 100:
                random.shuffle(self.dataset)
                prop = int(len(self.dataset) * self.args.train_proportion / 100)+1
                self.dataset = self.dataset[:prop]


        # if self.args.low_data == 'navi':
        #     self.dataset_new = []
        #     add_list = []
        #     for idx, d in enumerate(self.dataset):
        #         if d['actype_output'] == 12:
        #             add_list.append(d)
        #             self.dataset_new.append(d)
        #         elif d['actype_output'] == 0 and random.random()>0.5:
        #             pass
        #         else:
        #             self.dataset_new.append(d)
        #     self.dataset = self.dataset_new + add_list


        print('%s: %s action data instance: #%d'%(self.split, self.level, len(self)))
        return self.dataset

    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, idx):
        # get a data instance
        args = self.args

        instance = self.dataset[idx]

        # record the item in which data are transformed to PyTorch Tensors
        item = {'batch_type': self.task_type}
        item['interact'] = instance['interact'] if 'interact' in instance else 0
        for i in ['path', 'high_idx', 'low_idx']:
            if i in instance:
                item[i] = instance[i]

        # process the visual input
        task_path = instance['path']
        ins_obj_det_res = self.obj_det_res[task_path+str(instance['vision_input'])]

        vis_len = args.topk_objs + 1
        obj_num = min(vis_len - 1, len(ins_obj_det_res['score']))


        vis_feat_col = np.zeros((vis_len, 7))   # vis_len x 7
        vis_cls_col = np.zeros((vis_len, ), dtype=int)   # vis_len
        if not args.disable_feat_vis:
            for obj_idx in range(obj_num):
                bbox = [i/self.image_size for i in ins_obj_det_res['bbox'][obj_idx]]
                cls_idx = self.action_vocab.w2id(ins_obj_det_res['class'][obj_idx])
                vis_feat_col[obj_idx+1][:4] = bbox
                vis_feat_col[obj_idx+1][4] = bbox[2]-bbox[0]
                vis_feat_col[obj_idx+1][5] = bbox[3]-bbox[1]
                vis_feat_col[obj_idx+1][6] = ins_obj_det_res['score'][obj_idx]   # 1d
                vis_cls_col[obj_idx + 1] = cls_idx
            vis_cls_col[0] = 1
        item['vision_feats'] = vis_feat_col.astype(np.float32)   # np: vis_len x 7
        item['vision_cls'] = vis_cls_col   # np: vis_len

        # history visual input
        if self.args.enable_feat_vis_his and 'navi' in self.task_type:
            his_len = args.history_max_length - 1
            max_obj_num = 10
            his_vis_feat = np.zeros((his_len, max_obj_num, 7), dtype=np.float32)   # his_len x max_obj_num x 7
            his_vis_cls = np.zeros((his_len, max_obj_num), dtype=int)   # his_len x max_obj_num
            for his_idx, img_idx in enumerate(instance['vis_history_input'][-his_len:]):
                his_ins_obj_det_res = self.obj_det_res[task_path+str(img_idx)]
                obj_num = min(max_obj_num, len(his_ins_obj_det_res['class']))
                for obj_idx in range(obj_num):
                    bbox = [i/self.image_size for i in his_ins_obj_det_res['bbox'][obj_idx]]
                    cls_idx = self.action_vocab.w2id(his_ins_obj_det_res['class'][obj_idx])
                    his_vis_feat[his_idx][obj_idx] = bbox + [bbox[2]-bbox[0], bbox[3]-bbox[1], his_ins_obj_det_res['score'][obj_idx]]
                    his_vis_cls[his_idx][obj_idx] = cls_idx
            item['his_vis_feat'] = his_vis_feat
            item['his_vis_cls'] = his_vis_cls


        # dp = os.path.join('../../moca', task_path.replace('full_2.1.0_pp', 'json_feat_2.1.0'), 'feat_conv.pt')
        # img_idx = self.img_map[task_path][str(instance['vision_input'])]
        # item['resnet18feat'] = self.resnet18_feats[dp][img_idx]

        # process mask selection label
        label = ins_obj_det_res['label']   # None or an obj index
        if label is None or instance['arg_output'] == 105:
            label = -1
        elif isinstance(label, int):
            label += 1
            if label >= vis_len:
                label = 0
        elif item['interact']:
            # leave the vis idx 0 as the indicator of not sure about how to ground
            label = 0
        item['mask_label'] = label   # int

        # process the language input

        lang_len = args.lang_max_length
        lang_widx = np.zeros((lang_len,), dtype=int)   # lang_len
        # if not self.args.use_bert:
        # lang_widx[:2] = self.data.vocab.w2id('[SEP]')
        if not args.disable_feat_lang:
            wids = instance['lang_input']
            if self.args.use_bert:
                wids = mmt_word_ids_to_bert_ids(wids[1:-1], self.data.vocab, self.data.bert_tknz)
            actual_len = min(lang_len, len(wids))
            if actual_len != len(wids):
                print('warning: %d truncated to %s'%(len(wids), lang_len))
            lang_widx[:actual_len] = wids[:actual_len]
            if not self.args.use_bert:
                lang_widx[lang_widx >= self.data.vocab.vocab_size] = 0

        item['lang_input'] = lang_widx   #np: lang_len
        # else:
        #     if self.level == 'high':
        #         item['lang_input'] = random.choice(self.language_data[task_path]['goal'])
        #     else:
        #         item['lang_input'] = random.choice(self.language_data[task_path]['instr'])[instance['high_idx']]

        # process history actions
        action_len = args.history_max_length * 2   #*2 because an action consists of a type and an arg
        action_seq = np.zeros((action_len,), dtype=int)   # max_his_len*2
        if not args.disable_feat_action_his:
            for aidx, a in enumerate(instance['actype_history_input'][-args.history_max_length:]):
                action_seq[aidx*2] = a
                action_seq[aidx*2+1] = instance['arg_history_input'][-args.history_max_length:][aidx]
        item['action_history_input'] = action_seq   #np: max_his_len*2

        # action type/arg labels
        item['actype_label'] = instance['actype_output']  #int
        item['arg_label'] = instance['arg_output']   #int
        if 'navi' in self.task_type:
            item['arg_label'] = -1   # do not predict arguments for non-interactable low actions

        # # process the sequence mask
        seq_mask = np.zeros((args.max_enc_length,), dtype=int) + 1   # max_enc_length

        offset = 1
        for l, seq in [(vis_len, vis_cls_col), (lang_len, lang_widx) , (action_len, action_seq)]:
            seq_mask[offset: offset+l] = (seq!=0).astype(int)
            offset += l
        assert offset == args.max_enc_length
        item['seq_mask'] = seq_mask   #np: max_enc_length
        item['arg_pos'] = seq_mask.nonzero()[0][-1]
        item['type_pos'] = item['arg_pos'] - 1

        if 'navi' in self.task_type:
            if self.args.auxiliary_loss_navi:
                item['visible_label'] = instance['visible']
                item['reached_label'] = instance['reached']
                item['progress_label'] = instance['progress']
            if self.args.enable_feat_posture:
                item['rotation'] =int(instance['rotation'])
                item['horizon'] = int(instance['horizon'])%12

        return item


class AlfredDataset(object):

    def __init__(self, args):
        self.args = args
        self.raw_path = args.raw_data
        self.pp_path = args.pp_data   # preprocessed data saving path
        self.image_size = args.image_size
        with open(self.args.splits) as f:
            self.dataset_splits = json.load(f)
            pprint.pprint({k: len(v) for k, v in self.dataset_splits.items()})

        if not os.path.isdir(self.pp_path):
            os.makedirs(self.pp_path)

        # load/construct vocabularies
        self.prepare_vocab()

        # preprocess data
        if args.preprocess:
            if not args.skip_detection:
                # load trajectory images recorded in the full dataset
                self.image_hdf_path = args.img_data
                self.image_data = h5py.File(self.image_hdf_path, 'r')
                # self.image_data.visit(lambda x: print(x))
                self.mrcnn = MaskRCNNDetector(args, ['all', 'sep'])
                self.batch_size = 10
            self.init_statistics()
            self.preprocess_data()
            self.prepare_data_instances()
            self.save_statistics()


    def init_statistics(self):
        self.stats = {
            'goal length': Counter(),
            'instr length': Counter(),
            'high action steps': Counter(),
            'low action steps': Counter(),
            'detection num': Counter(),
            'object num': Counter(),
            'receptacle num': Counter(),
        }
        self.interact_num = 0
        self.good_detect_num = {'all':0, 'sep':0}


    def save_statistics(self):
        for k,v in self.stats.items():
            if isinstance(v, dict):
                self.stats[k] = dict(sorted(v.items(), key=lambda item: item[0]))
        with open(os.path.join(self.pp_path, 'statistics.json'), 'w') as f:
            json.dump(self.stats, f, indent=2)
        print('interact_num:', int(self.interact_num/2))
        print('good_detect_num:', self.good_detect_num)


    def prepare_vocab(self):
        # vocab save/load paths
        self.language_vocab_save = os.path.join(self.pp_path, 'vocab')
        self.dec_in_vocab_save = os.path.join(self.pp_path, 'dec_in_vocab')
        self.dec_out_vocab_high_save = os.path.join(self.pp_path, 'dec_out_vocab_high')
        self.dec_out_vocab_low_save = os.path.join(self.pp_path, 'dec_out_vocab_low')
        self.dec_out_vocab_args_save = os.path.join(self.pp_path, 'dec_out_vocab_arg')

        preprocess_vocab = not os.path.exists(self.language_vocab_save+'.w2id.json')
        # preprocess_vocab= True
        if preprocess_vocab:
            # natural language vocabulary (word <-> idx for encoder)
            self.vocab = Vocab(self.args.vocab_size, special_tokens=[PAD, UNK, SEP, SOS, 'None'])
            print('Constructing vocabulary for natural language')
            for k, d in self.dataset_splits.items():
                if 'test' in k:
                    continue # should not see test sets even for vocabulary construction
                print(' - dataset: {}'.format(k))
                for task in tqdm(d):
                    # load json file
                    json_path = os.path.join(self.args.raw_data, k, task['task'], 'traj_data.json')
                    with open(json_path) as f:
                        traj_raw = json.load(f)
                    self.process_language(traj_raw, {}, 0, for_vocab_construction=True)
            # save vocab in data path
            self.vocab.construct(self.language_vocab_save)

            print('Constructing vocabularies for encoder/decoder actions and objects')
            # decoder input (action/object names <-> idx)
            task_tokens = [PAD, UNK, SOS]+list(ACTION_TO_WORDS.keys())
            task_tokens += ALL_OBJECTS+['None']
            self.dec_in_vocab = Vocab(special_tokens=task_tokens)
            self.dec_in_vocab.construct(self.dec_in_vocab_save)

            # high-level decoder action output (high actions <-> idx)
            self.dec_out_vocab_high = Vocab(special_tokens=HIGH_ACTIONS)
            self.dec_out_vocab_high.construct(self.dec_out_vocab_high_save)

            # low-level decoder action output (high actions <-> idx)
            self.dec_out_vocab_low = Vocab(special_tokens=LOW_ACTIONS)
            self.dec_out_vocab_low.construct(self.dec_out_vocab_low_save)

            # decoder arguments output (object names <-> idx)
            self.dec_out_vocab_arg = Vocab(special_tokens=ACTION_ARGS)
            self.dec_out_vocab_arg.construct(self.dec_out_vocab_args_save)
        else:
            print('Loading vocabularies')
            self.vocab = Vocab()
            self.vocab.load(self.language_vocab_save, self.args.vocab_size)
            self.dec_in_vocab = Vocab()
            self.dec_in_vocab.load(self.dec_in_vocab_save)
            self.dec_out_vocab_high = Vocab()
            self.dec_out_vocab_high.load(self.dec_out_vocab_high_save)
            self.dec_out_vocab_low = Vocab()
            self.dec_out_vocab_low.load(self.dec_out_vocab_low_save)
            self.dec_out_vocab_arg = Vocab()
            self.dec_out_vocab_arg.load(self.dec_out_vocab_args_save)

            if self.args.use_bert:
                self.bert_tknz = get_bert_tknz(self.args)


    def preprocess_data(self):
        '''
        saves preprocessed data as jsons in specified folder
        '''
        if self.args.num_threads in [0,1]:
            for k, d in self.dataset_splits.items():
                print('Preprocessing {}'.format(k))
                # debugging:
                if self.args.fast_epoch:
                    d = d[:10]
                for task in tqdm(d):
                    self.preprocess_traj(k, task)
        else:
            task_queue = queue.Queue()
            for k, d in self.dataset_splits.items():
                if 'tests' in k:
                    continue
                if self.args.fast_epoch:
                    d = d[:30]
                for task in d:
                    task_queue.put((k, task))

            pbar = tqdm(total=task_queue.qsize())
            # start threads
            threads = []
            for n in range(self.args.num_threads):
                thread = threading.Thread(target=run, args=(self.preprocess_traj, task_queue, pbar))
                threads.append(thread)
                thread.start()
            for t in threads:
                t.join()

    def preprocess_traj(self, k, task):
        train_mode = 'test' not in k

        # load json file
        json_path = os.path.join(self.args.raw_data, k, task['task'], 'traj_data.json')
        with open(json_path) as f:
            traj_raw = json.load(f)

        # check if preprocessing storage folder exists
        pp_save_path = os.path.join(self.pp_path, k, task['task'])
        if not os.path.isdir(pp_save_path):
            os.makedirs(pp_save_path)

        traj_pp = {}
        # root & split
        traj_pp['raw_path'] = os.path.join(self.raw_path, k, task['task'])
        traj_pp['pp_path'] = pp_save_path
        traj_pp['split'] = k
        traj_pp['repeat_idx'] = r_idx = task['repeat_idx'] # index of the annotation for each trajectory

        # preprocess language
        self.process_language(traj_raw, traj_pp, r_idx)

        # for train/valid splits only
        if train_mode:
            self.process_actions(traj_raw, traj_pp)
            self.process_images(traj_raw, traj_pp)

        # save preprocessed json
        pp_json_path = os.path.join(pp_save_path, "ann_%d.json" % r_idx)
        with open(pp_json_path, 'w') as f:
            json.dump(traj_pp, f, indent=2)


    def tokenize(self, lang):
        return [SEP] + revtok.tokenize(remove_spaces_and_lower(lang)) + [SEP]


    def process_language(self, ex, traj, r_idx, for_vocab_construction=False):
        # goal instruction
        task_desc = ex['turk_annotations']['anns'][r_idx]['task_desc']

        # step-by-step instructions
        high_descs = ex['turk_annotations']['anns'][r_idx]['high_descs']

        # tokenize language
        traj['lang'] = {
            'repeat_idx': r_idx,
            'goal_tokenize': self.tokenize(task_desc),
            'instr_tokenize': [self.tokenize(x) for x in high_descs]
        }


        if for_vocab_construction:
            # add to vocab
            for w in traj['lang']['goal_tokenize']:
                self.vocab.add_word_counts(w)
            for instr in traj['lang']['instr_tokenize']:
                for w in instr:
                    self.vocab.add_word_counts(w)
            return

        self.stats['goal length'][len(traj['lang']['goal_tokenize'])] += 1
        for instr in traj['lang']['instr_tokenize']:
            self.stats['instr length'][len(instr)] += 1

        # word2idx
        traj['lang']['goal'] = self.vocab.seq_encode(traj['lang']['goal_tokenize'])
        traj['lang']['instr'] = [self.vocab.seq_encode(x) for x in traj['lang']['instr_tokenize']]


    def process_actions(self, ex, traj):

        def get_normalized_arg(a, level):
            if level == 'high':
                arg = a['discrete_action']['args']

                if arg == [] or arg == ['']:
                    return 'None'
                else:
                    arg =arg[-1]   #argument for action PutObject is the receptacle (2nd item in the list)
            elif level == 'low':
                if 'objectId' in a['api_action']:
                    if a['api_action']['action'] == 'PutObject':
                        arg = a['api_action']['receptacleObjectId'].split('|')[0]
                    elif len(a['api_action']['objectId'].split('|')) == 4:
                        arg = a['api_action']['objectId'].split('|')[0]
                    else:
                        arg = a['api_action']['objectId'].split('|')[4].split('_')[0]
                else:
                    return 'None'

            if arg in OBJECTS_LOWER_TO_UPPER:
                arg = OBJECTS_LOWER_TO_UPPER[arg]

            # fix high argument for sliced objects
            if level == 'high' and arg in {'Apple', 'Bread', 'Lettuce', 'Potato', 'Tomato'} and \
                'objectId' in a['planner_action'] and 'Sliced' in a['planner_action']['objectId']:
                arg += 'Sliced'
            return arg

        def fix_missing_high_pddl_end_action(ex):
            '''
            appends a terminal action to a sequence of high-level actions
            '''
            if ex['plan']['high_pddl'][-1]['planner_action']['action'] != 'End':
                ex['plan']['high_pddl'].append({
                    'discrete_action': {'action': 'NoOp', 'args': []},
                    'planner_action': {'value': 1, 'action': 'End'},
                    'high_idx': len(ex['plan']['high_pddl'])
                })

        # deal with missing end high-level action
        fix_missing_high_pddl_end_action(ex)

        # process high-level actions
        picked = None
        actions, args = [SOS], ['None']
        for idx, a in enumerate(ex['plan']['high_pddl']):
            high_action = a['discrete_action']['action']
            high_arg = get_normalized_arg(a, 'high')
            # change destinations into ones can be inferred
            # e.g. For task "clean a knife" turn GotoLocation(SideTable) to GotoLocation(Knife)
            if high_action == 'GotoLocation' and idx+1 < len(ex['plan']['high_pddl']):
                next_a = ex['plan']['high_pddl'][idx+1]
                if next_a['discrete_action']['action'] == 'PickupObject':
                    next_high_arg = get_normalized_arg(next_a, 'high')
                    high_arg = next_high_arg

            # fix argument of sliced object for Clean, Cool and Heat
            if high_action == 'PickupObject':
                picked = high_arg
            if high_action == 'PutObject':
                picked = None
            if picked is not None and 'Sliced' in picked and picked[:-6] == high_arg:
                high_arg = picked

            actions.append(high_action)
            args.append(high_arg)
        self.stats['high action steps'][len(actions)] += 1

        # high actions to action decoder input ids (including all task special tokens)
        traj['high'] = {}
        traj['high']['dec_in_high_actions'] = self.dec_in_vocab.seq_encode(actions)
        traj['high']['dec_in_high_args'] = self.dec_in_vocab.seq_encode(args)
        # high actions to high action decoder output ids
        traj['high']['dec_out_high_actions'] = self.dec_out_vocab_high.seq_encode(actions)[1:]
        traj['high']['dec_out_high_args'] = self.dec_out_vocab_arg.seq_encode(args)[1:]

        # process low-level actions
        num_hl_actions = len(ex['plan']['high_pddl'])
        # temporally aligned with HL actions
        traj['low'] = {}
        traj['low']['dec_in_low_actions'] = [list() for _ in range(num_hl_actions)]
        traj['low']['dec_in_low_args'] = [list() for _ in range(num_hl_actions)]
        traj['low']['dec_out_low_actions'] = [list() for _ in range(num_hl_actions)]
        traj['low']['dec_out_low_args'] = [list() for _ in range(num_hl_actions)]
        traj['low']['bbox'] = [list() for _ in range(num_hl_actions)]
        traj['low']['centroid'] = [list() for _ in range(num_hl_actions)]
        traj['low']['mask'] = [list() for _ in range(num_hl_actions)]
        traj['low']['interact'] = [list() for _ in range(num_hl_actions)]

        low_actions = [list() for _ in range(num_hl_actions)]
        low_args = [list() for _ in range(num_hl_actions)]
        prev_high_idx = -1
        for idx, a in enumerate(ex['plan']['low_actions']):
            # high-level action index (subgoals)
            high_idx = a['high_idx']

            if high_idx != prev_high_idx:
                # add NoOp to indicate the terimination of low-level action prediction
                low_actions[prev_high_idx].append('NoOp')
                low_args[prev_high_idx].append('None')
                # add the high-level action name as the first input of low-level action decoding
                high_action = ex['plan']['high_pddl'][high_idx]
                high_arg = get_normalized_arg(high_action, 'high')
                low_actions[high_idx].append(high_action['discrete_action']['action'])
                low_args[high_idx].append(high_arg)
                prev_high_idx = high_idx

            low_arg = get_normalized_arg(a, 'low')
            low_action = a['discrete_action']['action']
            if '_' in low_action:
                low_action = low_action.split('_')[0]
            low_actions[high_idx].append(low_action)
            low_args[high_idx].append(low_arg)

            # low-level bounding box (not used in the model)
            if 'bbox' in a['discrete_action']['args']:
                traj['low']['bbox'][high_idx].append(a['discrete_action']['args']['bbox'])
                xmin, ymin, xmax, ymax = [float(x) if x != 'NULL' else -1 for x in a['discrete_action']['args']['bbox']]
                traj['low']['centroid'][high_idx].append([
                    (xmin + (xmax - xmin) / 2) / self.image_size,
                    (ymin + (ymax - ymin) / 2) / self.image_size,
                    ])
            else:
                traj['low']['bbox'][high_idx].append([])
                traj['low']['centroid'][high_idx].append([])

            # low-level interaction mask (Note: this mask needs to be decompressed)
            mask = a['discrete_action']['args']['mask'] if 'mask' in a['discrete_action']['args'] else None
            traj['low']['mask'][high_idx].append(mask)

            # interaction validity
            has_interact = 0 if low_action in NON_INTERACT_ACTIONS else 1
            traj['low']['interact'][high_idx].append(has_interact)

        # add termination indicator for the last low-level action sequence
        low_actions[high_idx].append('NoOp')
        low_args[high_idx].append('None')

        for high_idx in range(num_hl_actions):
            actions, args = low_actions[high_idx], low_args[high_idx]
            traj['low']['dec_in_low_actions'][high_idx] = self.dec_in_vocab.seq_encode(actions)
            traj['low']['dec_in_low_args'][high_idx] = self.dec_in_vocab.seq_encode(args)
            traj['low']['dec_out_low_actions'][high_idx] = self.dec_out_vocab_low.seq_encode(actions)[1:]
            traj['low']['dec_out_low_args'][high_idx] = self.dec_out_vocab_arg.seq_encode(args)[1:]
            self.stats['low action steps'][len(actions)] += 1


        # check alignment between step-by-step language and action sequence segments
        action_low_seg_len = num_hl_actions
        lang_instr_seg_len = len(traj['lang']['instr'])
        seg_len_diff = action_low_seg_len - lang_instr_seg_len
        if seg_len_diff != 1:
            assert (seg_len_diff == 2) # sometimes the alignment is off by one  ¯\_(ツ)_/¯
            # print('Non align data file:', traj['raw_path'])
            # Because 1) this bug only in a few trajs 2) merge is very troublesome
            # we simply duplicate the last language instruction to align
            traj['lang']['instr_tokenize'].append(traj['lang']['instr_tokenize'][-1])
            traj['lang']['instr'].append(traj['lang']['instr'][-1])



    def process_images(self, ex, traj):
        def name2id(img_name):
            return int(img_name.split('.')[0])
        num_hl_actions = len(ex['plan']['high_pddl'])
        traj['high']['images'] = []
        traj['low']['images'] = [list() for _ in range(num_hl_actions)]

        prev_high_idx, prev_low_idx = -1, -1
        for img in ex['images']:
            high_idx, low_idx = img['high_idx'], img['low_idx']
            if high_idx != prev_high_idx:
                # reach a new high action, use the current image as the visual observation
                # 1) to predict the current high action
                # 2) to predict the termination low action of the previous high action
                traj['high']['images'].append(name2id(img['image_name']))
                if prev_high_idx >= 0:
                    traj['low']['images'][prev_high_idx].append(name2id(img['image_name']))
                prev_high_idx = high_idx
            if low_idx != prev_low_idx:
                # reach a new low action, use the current image as the visual observation
                # to predict the current low action
                traj['low']['images'][high_idx].append(name2id(img['image_name']))
                prev_low_idx = low_idx

        # add the last frame for predicting termination action NoOp
        traj['high']['images'].append(name2id(img['image_name']))
        traj['low']['images'][high_idx].append(name2id(img['image_name']))

        # length check
        assert(len(traj['high']['images']) == len(traj['high']['dec_in_high_actions']) - 1)
        for hi in range(num_hl_actions):
            assert(len(traj['low']['images'][hi]) == len(traj['low']['dec_in_low_actions'][hi]) - 1)

        # use mask rcnn for object detection
        if traj['repeat_idx'] != 0 or self.args.skip_detection:
            return   # for different annotations only need do object detection once

        all_imgs = [i for i in traj['high']['images']]
        for img_list in traj['low']['images']:
            all_imgs += img_list
        all_imgs = sorted(list(set(all_imgs)))

        dp = traj['raw_path'].replace('data/', '')
        results_masks, results_others = {}, {}

        for model_type in ['all', 'sep']:
            for idx in range(0, len(all_imgs), self.batch_size):
                batch = all_imgs[idx: idx + self.batch_size]
                img_path_batch = [self.image_data[dp][b] for b in batch]

                lock = threading.Lock()
                if model_type == 'all':
                    masks, boxes, classes, scores = self.mrcnn.get_mrcnn_preds_all(img_path_batch)
                else:
                    masks, boxes, classes, scores = self.mrcnn.get_mrcnn_preds_sep(img_path_batch)
                lock.release()

                # results_masks += masks
                for i, img_name in enumerate(batch):
                    results_others[img_name] = {
                        'bbox': [[int(coord) for coord in box] for box in boxes[i]],
                        'score': [float(s) for s in scores[i]],
                        'class': classes[i],
                        'label': None,
                    }
                    results_masks[img_name] = [np.packbits(m) for m in masks[i]]
                    self.stats['detection num'][len(classes[i])] += 1
                    self.stats['object num'][len([j for j in classes[i] if j in OBJECTS_DETECTOR])] += 1
                    self.stats['receptacle num'][len([j for j in classes[i] if j in STATIC_RECEPTACLES])] += 1

            # get object grounding labels
            for hidx, bbox_seq in enumerate(traj['low']['bbox']):
                for lidx, gt in enumerate(bbox_seq):
                    if gt:
                        self.interact_num += 1
                        img_idx = traj['low']['images'][hidx][lidx]
                        preds = results_others[img_idx]['bbox']
                        if not preds:
                            continue
                        max_iou = -1
                        for obj_idx, pred in enumerate(preds):
                            iou = bb_IoU(pred, gt)
                            if iou > max_iou:
                                max_iou = iou
                                best_obj_id, best_pred = obj_idx, pred
                        true_cls = ACTION_ARGS[traj['low']['dec_out_low_args'][hidx][lidx]]
                        try:
                            pred_cls = results_others[img_idx]['class'][best_obj_id]
                        except:
                            print('-'*10)
                            print('traj:', traj['raw_path'], 'img:', img_idx)
                        if max_iou > 0.7 or true_cls in pred_cls or pred_cls in true_cls:
                            results_others[img_idx]['label'] = best_obj_id
                            self.good_detect_num[model_type] += 1
                        # else:
                        #     print('-'*30)
                        #     print('traj:', traj['raw_path'], 'img:', img_idx)
                        #     print('iou: %.3f'%max_iou)
                        #     print('true class:', true_cls)
                        #     print('pred class:', pred_cls)
                        #     print('true:', gt)
                        #     print('pred:', best_pred)

            # save object detection results
            pp_save_path = traj['pp_path']
            pk_save_path = os.path.join(pp_save_path, "masks_%s.pkl"%model_type)
            json_save_path = os.path.join(pp_save_path, "bbox_cls_scores_%s.json"%model_type)
            with open(pk_save_path, 'wb') as f:
                pickle.dump(results_masks, f)
            with open(json_save_path, 'w') as f:
                json.dump(results_others, f, indent=4)


    def prepare_data_instances(self):
        statistic = {'horizon': Counter() ,'rotation': Counter(), 'mani': Counter(), 'navi': Counter()}
        mani_subgoal = {}
        for split, tasks in self.dataset_splits.items():
            if 'test' in split:
                continue
            print('Preparing %s data instances'%split)
            high_instances = []
            low_instances_mani = []
            low_instances_navi = []
            low_mani_seed = []
            low_navi_seed = []
            det_res_all = {}
            det_res_sep = {}

            cc=0
            for task in tqdm(tasks):
                # cc+=1
                # if cc == 30:
                #     break
                task_path = os.path.join(self.pp_path, split, task['task'])
                traj_path = os.path.join(task_path, 'ann_%d.json'%task['repeat_idx'])
                with open(traj_path, 'r') as f:
                    traj = json.load(f)
                with open(os.path.join(traj['raw_path'], 'traj_data.json'), 'r') as f:
                    traj_raw = json.load(f)
                init_action = traj_raw['scene']['init_action']

                obj_det_path_all = os.path.join(task_path, 'bbox_cls_scores_all.json')
                obj_det_path_sep = os.path.join(task_path, 'bbox_cls_scores_sep.json')
                with open(obj_det_path_all, 'r') as f:
                    obj_det_all = json.load(f)
                with open(obj_det_path_sep, 'r') as f:
                    obj_det_sep = json.load(f)
                for img_idx in obj_det_all:
                    det_res_all[task_path+img_idx] = obj_det_all[img_idx]
                    det_res_sep[task_path+img_idx] = obj_det_sep[img_idx]

                # process the vision input
                num_high_without_NoOp = len(traj['lang']['instr'])
                for hidx in range(num_high_without_NoOp + 1):
                    lang_input = traj['lang']['goal']   # list of int
                    vision_input = traj['high']['images'][hidx]
                    actype_history_input = traj['high']['dec_in_high_actions'][:hidx+1]
                    arg_history_input = traj['high']['dec_in_high_args'][:hidx+1]
                    actype_output = traj['high']['dec_out_high_actions'][hidx]
                    arg_output = traj['high']['dec_out_high_args'][hidx]
                    instance = {
                        'path': task_path,
                        'high_idx': hidx,
                        'lang_input': lang_input,
                        'vision_input': vision_input,
                        'actype_history_input': actype_history_input,
                        'arg_history_input': arg_history_input,
                        'actype_output': actype_output,
                        'arg_output': arg_output,
                    }
                    high_instances.append(instance)

                horizon = init_action['horizon']
                rotation = init_action['rotation']
                for hidx in range(num_high_without_NoOp):
                    # check the category of subgoal
                    sg_type = self.dec_in_vocab.id2w(traj['low']['dec_in_low_actions'][hidx][0])
                    sg_arg = self.dec_in_vocab.id2w(traj['low']['dec_in_low_args'][hidx][0])
                    subgoal = '%s(%s)'%(sg_type, sg_arg)
                    low_action_seq = ' '.join(self.dec_in_vocab.seq_decode(traj['low']['dec_in_low_actions'][hidx][1:]))
                    add_to_mani = split == 'train' and sg_type != 'GotoLocation' and subgoal not in statistic['mani']
                    add_to_navi = split == 'train' and sg_type == 'GotoLocation' and subgoal not in statistic['navi']
                    if add_to_mani:
                        if sg_type not in mani_subgoal:
                            mani_subgoal[sg_type] = Counter()
                        mani_subgoal[sg_type][low_action_seq] += 1
                        if mani_subgoal[sg_type][low_action_seq] == 1:
                            mani_subgoal[sg_type][low_action_seq+' (objs)'] = [sg_arg]
                        elif sg_arg not in mani_subgoal[sg_type][low_action_seq+' (objs)']:
                            mani_subgoal[sg_type][low_action_seq+' (objs)'].append(sg_arg)

                    lang_input = traj['lang']['instr'][hidx]
                    num_low_steps = len(traj['low']['dec_out_low_actions'][hidx])
                    vis_history = []
                    for low_idx in range(num_low_steps):
                        vision_input = traj['low']['images'][hidx][low_idx]
                        actype_history_input = traj['low']['dec_in_low_actions'][hidx][:low_idx+1]
                        arg_history_input = traj['low']['dec_in_low_args'][hidx][:low_idx+1]
                        actype_output = traj['low']['dec_out_low_actions'][hidx][low_idx]
                        arg_output = traj['low']['dec_out_low_args'][hidx][low_idx]
                        try:
                            interact = traj['low']['interact'][hidx][low_idx]
                        except:
                            interact = 0
                        if actype_history_input[0] == 3: # gotolocation
                            target_obj = self.dec_in_vocab.id2w(arg_history_input[0])
                            detected_objs = obj_det_sep[str(vision_input)]['class']
                            visible = 0 if target_obj not in detected_objs else 1
                            reached = 0
                            if low_idx == (num_low_steps - 1):
                                reached = 1
                            if low_idx == (num_low_steps - 2):
                                action = self.dec_out_vocab_low.id2w(traj['low']['dec_out_low_actions'][hidx][low_idx])
                                if action in {'LookDown', 'LookUp'} and visible:
                                    reached = 1
                            progress = (low_idx+1)/num_low_steps
                        else:
                            visible, reached, progress = -1, -1, -1

                        instance = {
                            'path': task_path,
                            'high_idx': hidx,
                            'low_idx': low_idx,
                            'interact': interact,
                            'lang_input': lang_input,
                            'vision_input': vision_input,
                            'actype_history_input': actype_history_input,
                            'arg_history_input': arg_history_input,
                            'vis_history_input': copy.deepcopy(vis_history),
                            'actype_output': actype_output,
                            'arg_output': arg_output,
                            'visible': visible,
                            'reached': reached,
                            'progress': progress,
                            'rotation': (rotation%360)/90,
                            'horizon': horizon/15,
                        }
                        statistic['horizon'][horizon] += 1
                        statistic['rotation'][rotation] += 1
                        if self.dec_out_vocab_low.id2w(actype_output) == 'RotateRight':
                            rotation +=  90
                        elif self.dec_out_vocab_low.id2w(actype_output) == 'RotateLeft':
                            rotation -=  90
                        elif self.dec_out_vocab_low.id2w(actype_output) == 'LookUp':
                            horizon +=  15
                        elif self.dec_out_vocab_low.id2w(actype_output) == 'LookDown':
                            horizon -= 15
                        vis_history.append(vision_input)

                        if actype_history_input[0] == 3: # gotolocation
                            low_instances_navi.append(instance)
                        else:
                            low_instances_mani.append(instance)

                        if add_to_mani:
                            low_mani_seed.append(instance)
                        if add_to_navi:
                            low_navi_seed.append(instance)

                    statistic['mani'][subgoal] += 1
                    statistic['navi'][subgoal] += 1

            print('high len:', len(high_instances))
            print('low mani len:', len(low_instances_mani))
            print('low navi len:', len(low_instances_navi))
            statistic['%s high len'%split] = len(high_instances)
            statistic['%s low-mani len'%split] = len(low_instances_mani)
            statistic['%s low-navi len'%split] = len(low_instances_navi)
            if split == 'train':
                statistic['train low-navi seed len'] = len(low_navi_seed)
                statistic['train low-mani seed len'] = len(low_mani_seed)

            high_save_path = os.path.join(self.pp_path, '%s_high_action_instances.json'%split)
            with open(high_save_path, 'w') as f:
                json.dump(high_instances, f, indent=2)
            low_save_path = os.path.join(self.pp_path, '%s_low_action_instances_mani.json'%split)
            with open(low_save_path, 'w') as f:
                json.dump(low_instances_mani, f, indent=2)
            with open(low_save_path.replace('mani', 'navi'), 'w') as f:
                json.dump(low_instances_navi, f, indent=2)

            if split == 'train':
                low_save_path = os.path.join(self.pp_path, '%s_low_action_seed_mani.json'%split)
                with open(low_save_path, 'w') as f:
                    json.dump(low_mani_seed, f, indent=2)
                with open(low_save_path.replace('mani', 'navi'), 'w') as f:
                    json.dump(low_navi_seed, f, indent=2)
                with open(os.path.join(self.pp_path, 'mani_subgoals.json'), 'w') as f:
                    json.dump(mani_subgoal, f, indent=2)

            det_all_save_path = os.path.join(self.pp_path, '%s_det_res_all.json'%split)
            with open(det_all_save_path, 'w') as f:
                json.dump(det_res_all, f, indent=2)
            det_sep_save_path = os.path.join(self.pp_path, '%s_det_res_sep.json'%split)
            with open(det_sep_save_path, 'w') as f:
                json.dump(det_res_sep, f, indent=2)

        for k,v in statistic.items():
            if isinstance(v, dict):
                statistic[k] = dict(sorted(v.items(), key=lambda item: item[0]))
        with open(os.path.join(self.pp_path, 'data_statistics.json'), 'w') as f:
            json.dump(statistic, f, indent=2)


def run(func, task_queue, pbar):
    while task_queue.qsize() > 0:
        lock.acquire()
        try:
            k, task = task_queue.get(False)
        except:
            lock.release()
            return
        lock.release()
        func(k,task)
        pbar.update(1)



if __name__ == '__main__':
    from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
    # os.system("taskset -p 0xffffffff %d" % os.getpid())
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    Config(parser)
    parser.add_argument('--skip_detection', action='store_true')
    args = parser.parse_args()
    pprint.pprint(args)
    dataset = AlfredDataset(args)
    # dataset.prepare_data_instances()