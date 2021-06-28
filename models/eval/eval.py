import os
import json
import pprint
import random
import time
import torch
import torch.multiprocessing as mp
from data.preprocess import Dataset
from importlib import import_module
# import threading


class EvalMMT(object):
    """iTHOR-based interactive evaluation

    Based on eval.py form the original Alfred repository.
    Modified to fit the proposed MMT model
    """
    def __init__(self, args, dataset, model, detector, multiproc_manager):
        # args and manager
        self.args = args
        self.manager = multiproc_manager   #run iThor env in parallel
        if '_' in self.args.eval_split:
            self.eval_splits = [self.args.eval_split]
        elif self.args.eval_split == 'valid':
            self.eval_splits = ['valid_seen', 'valid_unseen']
        elif self.args.eval_split == 'test':
            self.eval_splits = ['tests_seen', 'tests_unseen']

        # setup Alfred dataset and get vocabularies
        self.dataset = dataset
        self.pp_path = dataset.pp_path
        self.vocabs = {}
        self.vocabs['in_vocab_action'] = self.dataset.dec_in_vocab
        self.vocabs['out_vocab_high_type'] = self.dataset.dec_out_vocab_high
        self.vocabs['out_vocab_low_type'] = self.dataset.dec_out_vocab_low
        self.vocabs['out_vocab_arg'] = self.dataset.dec_out_vocab_arg

        # setup the model to be evaluated
        self.model = model
        if isinstance(model, dict):
            for m in model.values():
                m.share_memory()
                m.eval()
        else:
            self.model.share_memory()
            self.model.eval()

        # setup the object detector
        self.detector = detector
        self.detector.share_memory()
        self.detector.eval()

        # prepare the task directories for evaluation
        self.queue_tasks()


    def queue_tasks(self):
        self.stats = self.manager.dict()
        self.results = self.manager.dict()

        # queue tasks
        self.task_queue = self.manager.Queue()
        for split in self.eval_splits:
            # random.shuffle(self.dataset.dataset_splits[split])
            for task in self.dataset.dataset_splits[split]:
                task_path = os.path.join(self.pp_path, split, task['task'])
                self.task_queue.put((task_path, task['repeat_idx']))
                if self.args.fast_epoch and self.task_queue.qsize() == 3:
                    break
            else:
                continue
            break
        self.stats['task_num'] = self.task_queue.qsize()
        print('Total task num:', self.task_queue.qsize())


    def spawn_threads(self):
        '''
        spawn multiple threads to run eval in parallel
        '''
        self.stats['start_time'] = time.time()

        # start threads
        threads = []
        lock = self.manager.Lock()
        for n in range(self.args.num_threads):
            thread = mp.Process(target=self.run, args=(n, self.model, self.detector,
                self.vocabs, self.task_queue, self.args, lock, self.stats, self.results))
            thread.start()
            threads.append(thread)

        for t in threads:
            t.join()

        # self.save_results()


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

        # setup task for reward
        env.set_task(traj_data, args, reward_type=reward_type)


    @classmethod
    def run(cls, model, detector, vocabs, task_queue, args, lock, stats, results):
        raise NotImplementedError()

    @classmethod
    def evaluate(cls, env, model, detector, vocabs, traj_data, args, lock, stats, results):
        raise NotImplementedError()

    def save_results(self):
        raise NotImplementedError()

    def create_stats(self):
        raise NotImplementedError()

    def start(self):
        raise NotImplementedError()


class Eval(object):

    # tokens
    STOP_TOKEN = "<<stop>>"
    SEQ_TOKEN = "<<seg>>"
    TERMINAL_TOKENS = [STOP_TOKEN, SEQ_TOKEN]

    def __init__(self, args, manager):
        # args and manager
        self.args = args
        self.manager = manager

        # load splits
        with open(self.args.splits) as f:
            self.splits = json.load(f)
            pprint.pprint({k: len(v) for k, v in self.splits.items()})

        # load model
        print("Loading: ", self.args.model_path)
        M = import_module(self.args.model)
        self.model, optimizer = M.Module.load(self.args.model_path, self.args)
        self.model.share_memory()
        self.model.eval()
        self.model.test_mode = True

        # updated args
        self.model.args.dout = self.args.model_path.replace(self.args.model_path.split('/')[-1], '')
        self.model.args.data = self.args.data if self.args.data else self.model.args.data

        # preprocess and save
        if args.preprocess:
            print("\nPreprocessing dataset and saving to %s folders ... This is will take a while. Do this once as required:" % self.model.args.pp_folder)
            self.model.args.fast_epoch = self.args.fast_epoch
            dataset = Dataset(self.model.args, self.model.vocab)
            dataset.preprocess_splits(self.splits)

        # load resnet
        args.visual_model = 'resnet18'
        self.resnet = Resnet(args, eval=True, share_memory=True, use_conv_feat=True)

        # gpu
        if self.args.gpu:
            self.model = self.model.to(torch.device('cuda'))

        # success and failure lists
        self.create_stats()

        # set random seed for shuffling
        random.seed(int(time.time()))

    def queue_tasks(self):
        '''
        create queue of trajectories to be evaluated
        '''
        task_queue = self.manager.Queue()
        files = self.splits[self.args.eval_split]

        # debugging: fast epoch
        if self.args.fast_epoch:
            files = files[:16]

        if self.args.shuffle:
            random.shuffle(files)
        for traj in files:
            task_queue.put(traj)
        return task_queue

    def spawn_threads(self):
        '''
        spawn multiple threads to run eval in parallel
        '''
        task_queue = self.queue_tasks()

        # start threads
        threads = []
        lock = self.manager.Lock()
        for n in range(self.args.num_threads):
            thread = mp.Process(target=self.run, args=(self.model, self.resnet, task_queue, self.args, lock,
                                                       self.successes, self.failures, self.results))
            thread.start()
            threads.append(thread)

        for t in threads:
            t.join()

        # lock = threading.Lock()
        # for n in range(self.args.num_threads):
        #     thread = threading.Thread(target=self.run, args=(self.model, self.resnet, task_queue, self.args, lock, self.successes, self.failures, self.results))
        #     threads.append(thread)
        #     thread.start()
        #     time.sleep(1)

        # save
        self.save_results()

    @classmethod
    def setup_scene(cls, env, traj_data, r_idx, args, reward_type='dense'):
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

        # print goal instr
        print("Task: %s" % (traj_data['turk_annotations']['anns'][r_idx]['task_desc']))

        # setup task for reward
        env.set_task(traj_data, args, reward_type=reward_type)

    @classmethod
    def run(cls, model, resnet, task_queue, args, lock, successes, failures):
        raise NotImplementedError()

    @classmethod
    def evaluate(cls, env, model, r_idx, resnet, traj_data, args, lock, successes, failures):
        raise NotImplementedError()

    def save_results(self):
        raise NotImplementedError()

    def create_stats(self):
        raise NotImplementedError()

