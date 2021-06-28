import os, sys, argparse, json, copy, logging, time
from datetime import datetime
import torch
import torch.multiprocessing as mp

sys.path.append(os.path.join(os.environ['ALFRED_ROOT']))

from data.dataset import AlfredDataset
from models.config.configs import Config
from models.model.mmt import MultiModalTransformer
from models.nn.mrcnn import MaskRCNNDetector
from models.eval.eval import EvalMMT

from gen.constants import *
from env.thor_env import ThorEnv


class Leaderboard(EvalMMT):
    """iTHOR-based interactive evaluation

    Based on eval_task.py form the original Alfred repository.
    Modified to fit the proposed MMT model
    """
    def start(self):
        self.seen_actseqs, self.unseen_actseqs = self.manager.list(), self.manager.list()
        self.stats = self.manager.dict()
        self.stats['start_time'] = time.time()
        self.stats['task_num'] = self.task_queue.qsize()

        # for v in self.task_done['tests_seen']:
        #     self.seen_actseqs.append(v)
        # for v in self.task_done['tests_unseen']:
        #     self.unseen_actseqs.append(v)

        self.spawn_threads()
        Leaderboard.save_results(self.args, self.seen_actseqs, self.unseen_actseqs)

    # def queue_tasks(self):
    #     self.stats = self.manager.dict()
    #     self.results = self.manager.dict()

    #     # queue tasks
    #     self.task_queue = self.manager.Queue()


    #     with open(os.path.join(self.args.eval_path, 'test_max1_leaderboard_maxfail1_haspos_', 'tests_actseqs_dump_20210130_000035_085396.json'), 'r') as f:
    #         self.task_done = json.load(f)

    #     task_count = {}
    #     for i in self.task_done['tests_seen'] + self.task_done['tests_unseen']:
    #         (k, v), = i.items()
    #         if k in task_count:
    #             task_count[k] += 1
    #         else:
    #             task_count[k] = 1

    #     for split in ['tests_seen', 'tests_unseen']:
    #         for idx, task in enumerate(self.dataset.dataset_splits[split]):
    #             task_path = os.path.join(self.pp_path, split, task['task'])
    #             if task['task'] not in task_count or (task['repeat_idx']+1)>task_count[task['task']]:
    #                 self.task_queue.put((task_path, task['repeat_idx']))
    #     print('Total task num:', self.task_queue.qsize())

    @classmethod
    def save_results(cls, args, seen_actseqs, unseen_actseqs):
        '''
        save actseqs as JSONs
        '''
        results = {'tests_seen': list(seen_actseqs),
                        'tests_unseen': list(unseen_actseqs)}
        save_path = args.log_dir.replace('.log', '')
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        save_path = os.path.join(save_path, 'tests_actseqs_dump_' + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + '.json')
        with open(save_path, 'w') as r:
            json.dump(results, r, indent=4, sort_keys=True)


    def spawn_threads(self):
        '''
        spawn multiple threads to run eval in parallel
        '''

        # start threads
        threads = []
        lock = self.manager.Lock()
        # self.model.test_mode = True
        for n in range(self.args.num_threads):
            thread = mp.Process(target=self.run, args=(n, self.model, self.detector,
                self.vocabs, self.task_queue, self.args, lock, self.seen_actseqs, self.unseen_actseqs, self.stats))
            thread.start()
            threads.append(thread)

        for t in threads:
            t.join()



    @classmethod
    def setup_scene(cls, env, traj_data, args, reward_type='dense'):
        '''
        intialize the scene and agent from the task info
        '''
        # scene setup
        scene_num = traj_data['scene']['scene_num']
        object_poses = traj_data['scene']['object_poses']
        dirty_and_empty = traj_data['scene']['dirty_and_empty']
        object_toggles = traj_data['scene']['object_toggles']

        scene_name = 'FloorPlan%d' % scene_num
        env.reset(scene_name)
        env.restore_scene(object_poses, object_toggles, dirty_and_empty)

        # initialize to start position
        env.step(dict(traj_data['scene']['init_action']))


    @classmethod
    def run(cls, n, model, detector, vocabs, task_queue, args, lock, seen_actseqs, unseen_actseqs, stats):
        '''
        evaluation loop
        '''
        # num_core_per_proc = args.num_core_per_proc
        # # core_mask_op = "taskset -p %s %d" %(hex((2**num_core_per_proc-1)<<(num_core_per_proc*i)), os.getpid())
        # cores = ','.join([str(i + num_core_per_proc*n) for i in range(num_core_per_proc)])
        # core_mask_op = "taskset -pc %s %d" %(cores, os.getpid())
        # os.system(core_mask_op)
        # print('start process: %d (pid: %d) op: %s'%(n, os.getpid(), core_mask_op))

        # start THOR
        env = ThorEnv()

        # set logger
        log_level = logging.DEBUG if args.debug else logging.INFO
        log_handlers = [logging.StreamHandler(), logging.FileHandler(args.log_dir)]
        logging.basicConfig(handlers=log_handlers, level=log_level,
            format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

        while True:
            if task_queue.qsize() == 0:
                break

            task_path, r_idx = task_queue.get()
            task_path = task_path.replace('full_2.1.0_pp', 'full_2.1.0')
            try:
                with open(os.path.join(task_path, 'traj_data.json'), 'r') as f:
                    traj = json.load(f)
                traj['path'] = task_path
                cls.evaluate(env, r_idx, model, detector, vocabs, traj, args, lock,
                    seen_actseqs, unseen_actseqs, stats)
            except Exception as e:
                import traceback
                traceback.print_exc()
                logging.info("Error: " + repr(e))
                # quit()

        # stop THOR
        env.stop()

    @classmethod
    def evaluate(cls, env, r_idx, model, detector, vocabs, traj_data, args, lock, seen_actseqs, unseen_actseqs, stats):

        prtfunc = logging.debug
        prtfunc('-'*50)
        prtfunc(traj_data['path'])

        if isinstance(model, dict):
            model_navi = model['navi']
            model_mani = model['mani']
        else:
            model_navi = model_mani = model

        # setup scene
        cls.setup_scene(env, traj_data, args, reward_type='dense')
        try:
            horizon = traj_data['scene']['init_action']['horizon']
            rotation = traj_data['scene']['init_action']['rotation']
        except:
            horizon = rotation = 0

        all_done, success = False, False
        high_idx, high_fails, high_steps = 0, 0, 0
        high_history = ['[SOS]', 'None']
        high_history_before_last_navi = copy.deepcopy(high_history)
        low_history_col = []
        api_fails = 0
        t = 0
        stop_cond = ''
        failed_events= ''
        terminate = False
        api_actions = []

        t_start_total = time.time()
        while not all_done:
            high_steps += 1
            # break if max_steps reached
            if high_fails >= args.max_high_fails:
                stop_cond += 'Terminate due to reach maximum high repetition failures'
                break
            # if high_fails >= 3 and high_idx in {0, 2}:
            #     stop_cond += 'Terminate due to failed attempts in early stage of navigation'
            #     break
            if high_idx >= args.max_high_steps:
                stop_cond += 'Terminate due to reach maximum high steps'
                break

            with torch.no_grad():
                curr_frame = env.last_event.frame
                masks, boxes, classes, scores = detector.get_preds_step(curr_frame)
                # prtfunc('MaskRCNN Top8: ' + ''.join(['(%s, %.3f)'%(i, j) for i, j in zip(classes[:8], scores[:8])]))

                observations = {}
                raw_lang = traj_data['turk_annotations']['anns'][r_idx]['task_desc']
                encoded_lang = model.dataset.vocab.seq_encode(model.dataset.tokenize(raw_lang))
                observations['lang'] = encoded_lang if not args.eval_disable_feat_lang_high else None
                observations['vis'] = [boxes, classes, scores] if not args.eval_disable_feat_vis else None
                observations['act_his'] = high_history if not args.eval_disable_feat_action_his else None
                observations['vis_his'] = None
                observations['pos'] = None

                # model predicts topk proposals
                preds, probs  = model_mani.step(observations, 'high', topk=5)
                high_actions, high_args = preds['type'], preds['arg']


                # Simply use the top1 high prediction instead of using multiple high proposals
                high_action = high_actions[0]
                high_arg = high_args[0]

                prtfunc('-'*50)
                prtfunc('Task goal: ' + traj_data['turk_annotations']['anns'][r_idx]['task_desc'])
                prtfunc('High proposals:')
                prtfunc('action: ' + ''.join(['(%s, %.3f)'%(high_abbr[i], j) for i, j in zip(high_actions, probs['type'])]))
                prtfunc('arg: ' + ''.join(['(%s, %.3f)'%(i, j) for i, j in zip(high_args, probs['arg'])]))

            if high_action == 'GotoLocation':
                high_history_before_last_navi = copy.deepcopy(high_history)


            if high_action == 'NoOp':
                all_done = True
                stop_cond += 'Predictes a high-level NoOp to terminate!'
                break

            # print action
            prtfunc('high history' + str(high_history))
            prtfunc('high pred: %s(%s)'%(high_action, high_arg))
            prtfunc('high idx: %d'%high_idx)
            prtfunc('high fails: %d'%high_fails)

            # go into the low-level action prediction loop
            subgoal_done, prev_t_success = False, False
            low_idx = 0
            low_history = [high_action, high_arg]
            low_vis_history = []
            while not subgoal_done:
                # break if max_steps reached
                if low_idx >= args.max_low_steps:
                    failed_events += 'SG %s(%s) not done in %d steps |'%(high_action, high_arg, args.max_low_steps)
                    prtfunc("Reach maximum low step limitation. Subgoal '%s(%s)' failed" %(high_action, high_arg))
                    break

                prtfunc('-'*50)
                prtfunc('Completing subgoal: %s(%s)'%(high_action, high_arg))

                with torch.no_grad():
                    task_type = 'low_navi' if high_action == 'GotoLocation' else 'low_mani'
                    model = model_navi if task_type == 'low_navi' else model_mani
                    # visual observation
                    curr_frame = env.last_event.frame
                    masks, boxes, classes, scores = detector.get_preds_step(curr_frame)
                    prtfunc('MaskRCNN Top8: ' + ''.join(['(%s, %.3f)'%(i, j) for i, j in zip(classes[:8], scores[:8])]))

                    # disable language directives when retry the navigation subgoal
                    global_disable = args.eval_disable_feat_lang_navi if 'navi' in task_type else args.eval_disable_feat_lang_mani
                    total_high_num = len(traj_data['turk_annotations']['anns'][r_idx]['high_descs'])
                    use_lang = not global_disable and high_idx == len(low_history_col) and high_idx<total_high_num
                    if use_lang:
                        prtfunc('[%d] Instr: '%(high_idx+1) + traj_data['turk_annotations']['anns'][r_idx]['high_descs'][high_idx])
                    use_vis_his = args.eval_enable_feat_vis_his and 'navi' in task_type
                    use_pos = args.eval_enable_feat_posture and 'navi' in task_type


                    observations = {}
                    if use_lang:
                        raw_lang = traj_data['turk_annotations']['anns'][r_idx]['high_descs'][high_idx]
                        encoded_lang = model.dataset.vocab.seq_encode(model.dataset.tokenize(raw_lang))
                        observations['lang'] = encoded_lang
                    else:
                        observations['lang'] = None
                    observations['vis'] = [boxes, classes, scores] if not args.eval_disable_feat_vis else None
                    observations['act_his'] = low_history if not args.eval_disable_feat_action_his else None
                    observations['vis_his'] = low_vis_history if use_vis_his else None
                    observations['pos'] = {'rotation': int((rotation%360)/90),
                                                        'horizon': int(horizon/15)%12} if use_pos else None

                    # model predicts topk proposals
                    preds, probs = model.step(observations, task_type, topk=13)
                    low_actions, low_args, mask_ids = preds['type'], preds['arg'], preds['mask']

                    if task_type == 'low_navi':
                        # proposals = [(low_actions[i], None, None) for i in range(5) if probs['type'][i]]
                        proposals = [(low_actions[0], None, None)]
                        # Obstruction Detection technique from MOCA
                        if low_actions[0] == 'MoveAhead':
                            cands = ['RotateLeft', 'RotateRight', 'NoOp']
                            cand_rank= [low_actions.index(i) for i in cands]
                            probs['type'][cand_rank[2]] += 0.05 # encourage stopping
                            cand_probs = [probs['type'][i] for i in cand_rank]
                            cand_idx = cand_probs.index(max(cand_probs))
                            proposals.append((cands[cand_idx], None, None))
                        prtfunc('Low proposal: '+ str([i for i,j,k in proposals]))
                    else:
                        # proposals = [(low_actions[0], low_args[i], mask_ids[i]) for i in range(3)]
                        proposals = []
                        for act_idx, act_prob in enumerate(probs['type']):
                            for arg_idx, arg_prob in enumerate(probs['type']):
                                if act_prob > 0.1 and arg_prob > 0.1:
                                    proposals.append((low_actions[act_idx], low_args[arg_idx], mask_ids[arg_idx]))
                                    prtfunc('Low proposal: %s(%.2f) %s(%.2f)'%(low_actions[act_idx], act_prob,
                                                                                                            low_args[arg_idx], arg_prob))

                # prtfunc('Low proposals:' + str(proposals))
                # prtfunc('action: ' + ''.join(['(%s, %.3f)'%(i, j) for i, j in zip(low_actions, probs['type'])]))
                # prtfunc('arg: ' + ''.join(['(%s, %.3f)'%(i, j) for i, j in zip(low_args, probs['arg'])]))
                # prtfunc('mask: ' + ''.join(['(%s, %.3f)'%(i, j) for i, j in zip(mask_ids, probs['mask'])]))

                t_success = False
                for action, low_arg, mask_id in proposals:
                    if action == 'NoOp' and (high_action=='GotoLocation' or prev_t_success):
                        low_history += [action, low_arg]
                        low_vis_history += [observations['vis']] if args.eval_enable_feat_vis_his else []
                        subgoal_done = True
                        prtfunc("Subgoal '%s(%s)' is done! low steps: %d"%(high_action, high_arg, low_idx+1))
                        break

                    # disable masks/arguments for non-ineractive actions
                    if action in NON_INTERACT_ACTIONS:
                        low_arg = 'None'
                        mask = None
                    else:
                        try:
                            mask = masks[mask_id - 1]
                            mask_cls_pred = classes[mask_id-1]
                            if not similar(mask_cls_pred, low_arg):
                                failed_events += 'bad mask: %s -> %s |'%(mask_cls_pred, low_arg)
                                prtfunc('Agent: no correct mask grounding!')
                                mask = None
                        except:
                            # if low_arg in classes:
                            #     mask = masks[classes.index(low_arg)]
                            if mask_id == 0:
                                failed_events += 'pred mask: 0 |'
                                prtfunc('Agent: no available mask')
                                mask = None
                            else:
                                failed_events += 'Invaild mask id: %s |'%str(mask_id)
                                prtfunc('Invaild mask id: %s'%str(mask_id))
                                mask = None

                    if action not in NON_INTERACT_ACTIONS and mask is None:
                        prev_t_success = False
                        continue

                    # use action and predicted mask (if available) to interact with the env
                    t_success, _, _, err, api_action = env.va_interact(action, interact_mask=mask,
                        smooth_nav=False, debug=args.debug)
                    t += 1
                    prev_t_success = t_success

                    if api_action is not None:
                        api_actions.append(api_action)

                    if t_success:
                        low_history += [action, low_arg]
                        low_vis_history += [observations['vis']] if args.eval_enable_feat_vis_his else []
                        rotation += 90 if action == 'RotateRight' else -90 if action == 'RotateLeft' else 0
                        horizon += 15 if action == 'LookUp' else -15 if action == 'LookDown' else 0
                        prtfunc('low pred: %s(%s)'%(action, low_arg))
                        prtfunc('Agent posture: rotation: %d horizon: %d'%(rotation, horizon))
                        prtfunc('high idx: %d (fails: %s)  low idx: %d'%(high_idx, high_fails, low_idx))
                        prtfunc('Successfully executed!')
                        prtfunc('Low history: '+' '.join(['%s'%i for i in low_history[::2]]))
                        low_idx += 1
                        break
                    else:
                        api_fails += 1
                        failed_events += 'bad action: %s(%s) api_fail: %d |'%(action, low_arg, api_fails)
                        # the ALFRED benchmark restriction, should not be changed!
                        if api_fails >= args.max_fails:
                            stop_cond = 'Reach 10 Api fails'
                            terminate = True
                        prtfunc('Low pred: %s(%s)'%(action, low_arg))
                        prtfunc('Low action failed! Try another low proposal')

                if terminate:
                    break

                if not prev_t_success:   # fails in all the proposals, get stuck
                    failed_events += 'SG %s(%s) fail(%d): no valid proposal |'%(high_action, high_arg, high_fails)
                    prtfunc("Failed in all low proposals. Subgoal '%s(%s)' failed" %(high_action, high_arg))
                    break

            # out of the low loop and return to the high loop
            if terminate:
                break

            if subgoal_done:
                if high_idx == len(low_history_col):
                    # a new subgoal is completed
                    low_history_col.append(low_history)
                    # high_fails = 0
                else:
                    # a previously failed subgoal is completed
                    low_history_col[high_idx].extend(low_history[2:])
                high_history += [high_action, high_arg]
                high_idx += 1
            else:
                high_fails += 1
                if high_action == 'GotoLocation':
                    # if a navigation subgoal fails, simply retry it
                    high_history = high_history
                    high_idx = high_idx
                    failed_events += 'Navi failed (step: %d) |'%high_steps
                    prtfunc("Navigation failed. Try again!")
                else:
                    # if a manipulative subgoal fails, retry the navigation subgoal before that
                    high_history = copy.deepcopy(high_history_before_last_navi)
                    high_idx = int(len(high_history)/2 - 1)
                    failed_events += 'SG %s(%s) failed: go back to hidx: %d |'%(high_action, high_arg, high_idx)
                    prtfunc("Subgoal '%s(%s)' failed. Retry the navigation before that!" %(high_action, high_arg))


        # End loop
        prtfunc('Stop: %s'%stop_cond)

        # actseq
        actseq = {traj_data['task_id']: api_actions}

        # log success/fails
        lock.acquire()
        if '_seen' in traj_data['path']:
            seen_actseqs.append(actseq)
        else:
            unseen_actseqs.append(actseq)

        logging.info("Task ID: %s (ridx: %d)" % (traj_data['task_id'], r_idx))

        for hidx in range(len(traj_data['turk_annotations']['anns'][r_idx]['high_descs'])):
            # try:
            logging.info('[%d] Instr: '%(hidx+1) + traj_data['turk_annotations']['anns'][r_idx]['high_descs'][hidx])
            try:
                logging.info('Pred High : %s(%s)'%(high_abbr[high_history[2:][2*hidx]], high_history[2:][2*hidx+1]))
                if low_history_col[hidx][2:][0] in navi_abbr:
                    logging.info('Prew Low: '+' '.join(['%s'%(navi_abbr.get(i, i)) for i in low_history_col[hidx][2:][::2]]))
                else:
                    logging.info('Pred Low: '+' '.join(['%s'%(i) for i in low_history_col[hidx][2:]]))
            except:
                logging.info('Reocrding failed. ')
                break
        logging.info('Failure records: \n%s'%failed_events.replace('|', '\n')[:-2])

        logging.info('Task goal: ' + traj_data['turk_annotations']['anns'][r_idx]['task_desc'])
        logging.info('High pred: '+' '.join(['%s(%s)'%(high_abbr[i], high_history[2*idx+3]) for idx, i in enumerate(high_history[2:][::2])]))
        logging.info('Fail num high %d | api: %d'%(high_fails, api_fails))
        logging.info('Stop condition: %s'%stop_cond)
        num_eval = len(seen_actseqs) + len(unseen_actseqs)
        elapsed = (time.time()- stats['start_time'])/60
        logging.info('%d / %d evaluated | elapsed: %.1f min '%(num_eval, stats['task_num'], elapsed))
        if num_eval %500 == 0:
            cls.save_results(args, seen_actseqs, unseen_actseqs)
        lock.release()



if __name__ == '__main__':
    core_mask_op = "taskset -pc %s %d" %('0-40', os.getpid())
    os.system(core_mask_op)

    # multiprocessing settings
    mp.set_start_method('spawn')
    manager = mp.Manager()

    # parser
    parser = argparse.ArgumentParser()
    Config(parser)

    # settings
    parser.add_argument('--eval_path', type=str, default="exp/something")
    parser.add_argument('--eval_split', type=str, default="test")
    parser.add_argument('--ckpt_name', type=str, default="model_best_valid.pth")
    parser.add_argument('--num_core_per_proc', type=int, default=5, help='cpu cores used per process')
    # parser.add_argument('--model', type=str, default='models.model.seq2seq_im_mask')
    parser.add_argument('--max_high_steps', type=int, default=20, help='max steps before a high-level episode termination')
    parser.add_argument('--max_high_fails', type=int, default=10, help='max failing times to try high-level proposals')
    parser.add_argument('--max_low_steps', type=int, default=50, help='max steps before a low-level episode termination')

    parser.add_argument('--eval_disable_feat_lang_high', help='do not use language features as high input', action='store_true')
    parser.add_argument('--eval_disable_feat_lang_navi', help='do not use language features as low-navi input', action='store_true')
    parser.add_argument('--eval_disable_feat_lang_mani', help='do not use language features as low-mani input', action='store_true')
    parser.add_argument('--eval_disable_feat_vis', help='do not use visual features as input', action='store_true')
    parser.add_argument('--eval_disable_feat_action_his', help='do not use action history features as input', action='store_true')
    parser.add_argument('--eval_enable_feat_vis_his', help='use additional history visual features as input', action='store_true')
    parser.add_argument('--eval_enable_feat_posture', help='use additional agent posture features as input', action='store_true')

    # parse arguments
    args = parser.parse_args()

    # fixed settings (DO NOT CHANGE)
    args.max_steps = 1000
    args.max_fails = 10

    # parse arguments
    args_model = argparse.Namespace(**json.load(open(os.path.join(args.eval_path, 'config.json'), 'r')))
    args.use_bert = args_model.use_bert
    args.bert_model = args_model.bert_model
    # args.inner_dim = 1024

    # load alfred data and build pytorch data sets and loaders
    alfred_data = AlfredDataset(args)


    # setup model
    device = torch.device('cuda') if args.gpu else torch.device('cpu')
    ckpt_path = os.path.join(args.eval_path, args.ckpt_name)
    ckpt = torch.load(ckpt_path, map_location=device)
    model = MultiModalTransformer(args_model, alfred_data)
    model.load_state_dict(ckpt, strict=False)   #strict=False
    model.to(model.device)
    models = model


    # log dir
    eval_type = 'leaderboard'
    input_str = ''
    if args.eval_disable_feat_lang_high:
        input_str += 'nolanghigh_'
    if args.eval_disable_feat_lang_mani:
        input_str += 'nolangmani_'
    if args.eval_disable_feat_lang_navi:
        input_str += 'nolangnavi_'
    if args.eval_disable_feat_vis:
        input_str += 'novis_'
    if args.eval_disable_feat_action_his:
        input_str += 'noah_'
    if args.eval_enable_feat_vis_his:
        input_str += 'hasvh_'
    if args.eval_enable_feat_posture:
        input_str += 'haspos_'
    log_name = '%s_%s_maxfail%d_%s.log'%(args.name_temp, eval_type, args.max_high_fails, input_str)
    if args.debug:
        log_name = log_name.replace('.log', '_debug.log')
    if isinstance(models, dict):
        log_name = log_name.replace('.log', '_sep.log')
    args.log_dir = os.path.join(args.eval_path, log_name)

    log_level = logging.DEBUG if args.debug else logging.INFO
    log_handlers = [logging.StreamHandler(), logging.FileHandler(args.log_dir)]
    logging.basicConfig(handlers=log_handlers, level=log_level,
        format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')


    # setup object detector
    detector = MaskRCNNDetector(args, detectors=[args.detector_type])


    # leaderboard dump
    eval = Leaderboard(args, alfred_data, models, detector, manager)

    # start threads
    eval.start()