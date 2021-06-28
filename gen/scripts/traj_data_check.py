import os,sys
import json
import pprint
import random
import time
import argparse
import pickle
import logging

import numpy as np
import torch
import torch.multiprocessing as mp

sys.path.append(os.path.join(os.environ['ALFRED_ROOT']))

from data.dataset import AlfredDataset
from models.config.configs import Config
from env.thor_env import ThorEnv
from gen.utils.image_util import decompress_mask
from models.nn.mrcnn import MaskRCNNDetector


# os.system("taskset -p 0xff %d" % os.getpid())
os.system("taskset -p 0xffffffff %d" % os.getpid())

class EvalExpertTrajs(object):

    def __init__(self, args, manager):
        # args and manager
        self.args = args
        self.manager = manager
        self.data = AlfredDataset(args)
        self.pp_path = args.pp_data
        self.action_vocab = self.data.dec_in_vocab

        # load splits
        with open(self.args.splits) as f:
            self.splits = json.load(f)

        # queue tasks
        args.task_num = 0
        self.task_queue = self.manager.Queue()
        for task in self.splits[args.eval_split]:
            if task['repeat_idx'] != 0:
                args.task_num += 1
                continue
            task_path = os.path.join(self.pp_path, args.eval_split, task['task'])
            self.task_queue.put(task_path)
            args.task_num += 1

        self.successes, self.failures = self.manager.list(), self.manager.list()
        self.avg_fps = self.manager.Value('i', 0)


    def spawn_threads(self):
        '''
        spawn multiple threads to run eval in parallel
        '''
        self.args.start_time = time.time()

        detector = MaskRCNNDetector(self.args, detectors=[args.detector_type])

        # start threads
        threads = []
        lock = self.manager.Lock()
        for n in range(self.args.num_threads):
            thread = mp.Process(target=self.run, args=(self.task_queue, self.args, self.action_vocab, lock,
                                                       self.successes, self.failures, self.avg_fps, detector))
            thread.start()
            threads.append(thread)

        for t in threads:
            t.join()

        self.save_results()


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
    def run(cls, task_queue, args, action_vocab, lock, successes, failures, avg_fps, detector):
        '''
        evaluation loop
        '''
        # start THOR
        env = ThorEnv()

        # set logger
        stderr_handler = logging.StreamHandler()
        file_handler = logging.FileHandler(args.save_name)
        logging.basicConfig(handlers=[stderr_handler, file_handler])
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        while True:
            if task_queue.qsize() == 0:
                break

            task_path = task_queue.get()

            try:
                traj_path = os.path.join(task_path, 'ann_0.json')
                with open(traj_path, 'r') as f:
                    traj = json.load(f)
                with open(os.path.join(traj['raw_path'], 'traj_data.json'), 'r') as f:
                    traj_raw = json.load(f)
                traj['raw'] =traj_raw   # key word 'plan' is used in tasks.py

                # obj_det_path = os.path.join(task_path, 'bbox_cls_scores_%s.json'%args.detector_type)
                # with open(obj_det_path, 'r') as f:
                #     obj_det_res = json.load(f)
                # mask_path = os.path.join(task_path, 'masks_%s.pkl'%args.detector_type)
                # with open(mask_path, 'rb') as f:
                #     masks = pickle.load(f)
                cls.evaluate(env, traj, None, None, args, action_vocab, lock, successes, failures, avg_fps, detector)
            except Exception as e:
                import traceback
                traceback.print_exc()
                print("Error: " + repr(e))

        # stop THOR
        env.stop()

    @classmethod
    def evaluate(cls, env, traj_data, obj_det_res, masks, args, action_vocab, lock, successes, failures, avg_fps, detector):
        cls.setup_scene(env, traj_data['raw'], args, reward_type='dense')

        low_actions, low_args, pred_masks, true_masks, interacts = [], [], [], [], []
        for hidx, low_action_seq in enumerate(traj_data['low']['dec_in_low_actions']):
            for low_idx, low_action_idx in enumerate(low_action_seq[1:-1]):
                # low_args.append(action_vocab.id2w(traj_data['low']['dec_in_low_args'][low_idx]))
                low_actions.append(action_vocab.id2w(low_action_idx))
                interact = traj_data['low']['interact'][hidx][low_idx]
                interacts.append(interact)
                img_idx = traj_data['low']['images'][hidx][low_idx]
                # pred_index = obj_det_res[str(img_idx)]['label']
                if not interact:
                    true_masks.append(None)
                    # pred_masks.append(None)
                else:
                    true_masks.append(decompress_mask(traj_data['low']['mask'][hidx][low_idx]))
                    # if interact and pred_index is not None:
                    #     # mask_load = np.unpackbits(masks[img_idx][pred_index])
                    #     mask_load = np.reshape(mask_load, (300,300)).astype(bool)
                    #     pred_masks.append(mask_load)
                    # else:
                    #     empty_mask = np.zeros((300,300), dtype=np.bool)
                    #     pred_masks.append(empty_mask)

        done, success = False, False
        fails = 0
        t = 0
        reward = 0

        t_start_total = time.time()

        for step_idx in range(len(low_actions)):
            # extract visual features

            # get action and mask
            action = low_actions[step_idx]
            true_mask = true_masks[step_idx]
            # pred_mask = pred_masks[step_idx]
            interact = interacts[step_idx]
            # mask = true_mask if args.mask_type=='true' else pred_mask
            curr_frame = env.last_event.frame

            if args.mask_type=='true':
                mask = true_mask
            elif true_mask is not None:
                # mask = np.squeeze(mask)
                masks, boxes, classes, scores = detector.get_preds_step(curr_frame)
                overlap = 0
                for msk in masks:
                    olp = np.sum((msk==1) & (true_mask==1)) / np.sum((msk==1) | (true_mask==1))
                    # print(olp)
                    if olp > overlap:
                        mask = msk
                        overlap = olp
                #         print(olp)
                # print(mask)
                # print(true_mask)
            else:
                mask = None

            # print action
            if args.debug:
                print('pred:', action)

            # use action and predicted mask (if available) to interact with the env
            t_success, _, _, err, _ = env.va_interact(action, interact_mask=mask, smooth_nav=args.smooth_nav, debug=args.debug)

            if not t_success:
                break

            # next time-step
            t_reward, t_done = env.get_transition_reward()
            reward += t_reward
            t += 1

        logging.info("-------------"*5)
        logging.info("Task ID: %s" % (traj_data['raw']['task_id']))
        logging.info("Task type: %s" % (traj_data['raw']['task_type']))
        total_time = time.time() - t_start_total
        fps = t / total_time
        logging.info('Time cost: %.1f sec, FPS: %.2f'%(total_time, fps))

        # check if goal was satisfied
        goal_satisfied = env.get_goal_satisfied()
        if goal_satisfied:
            success = True

        # goal_conditions
        pcs = env.get_goal_conditions_met()
        goal_condition_success_rate = pcs[0] / float(pcs[1])

        logging.info("Task succeed: %s"%str(success))
        logging.info("Goal conditions meet: %d / %d"%(pcs[0], pcs[1]))

        # log success/fails
        lock.acquire()
        log_entry = {'trial': traj_data['raw']['task_id'],
                     'type': traj_data['raw']['task_type'],
                     'completed_goal_conditions': int(pcs[0]),
                     'total_goal_conditions': int(pcs[1]),
                     'goal_condition_success': float(goal_condition_success_rate),
                     'reward': float(reward)}

        repeat_num = len(traj_data['raw']['turk_annotations']['anns'])

        if success:
            successes.extend([log_entry]*repeat_num)
        else:
            failures.extend([log_entry]*repeat_num)
        num_success = len(successes)
        num_failure = len(failures)
        num_eval = num_success+ num_failure
        avg_fps_new = (avg_fps.value * (num_eval - repeat_num) + fps * repeat_num) / num_eval
        avg_fps.value = avg_fps_new

        # overall results
        completed_goal_conditions = sum([entry['completed_goal_conditions'] for entry in successes]) + \
                                   sum([entry['completed_goal_conditions'] for entry in failures])
        total_goal_conditions = sum([entry['total_goal_conditions'] for entry in successes]) + \
                               sum([entry['total_goal_conditions'] for entry in failures])
        lock.release()

        prtfunc = print if num_eval != args.task_num else logging.info
        prtfunc('Eval result so far: ')
        if args.mask_type == 'true':
            mask_str = '[true mask] ' + args.eval_split
        elif args.detector_type == 'all':
            mask_str = '[pred all mask] ' + args.eval_split
        elif args.detector_type == 'sep':
            mask_str = '[pred sep mask] ' + args.eval_split
        elapsed = (time.time()- args.start_time)/60
        prtfunc('%s %d/%d evaluated | elapsed: %.1f min | avg fps: %.2f (x %d procs)'%(mask_str, num_eval,
            args.task_num, elapsed, avg_fps_new, args.num_threads))
        prtfunc("SR: %d/%d = %.3f" % (num_success,num_eval, num_success/num_eval))
        prtfunc("GC: %d/%d = %.3f" % (completed_goal_conditions,
                                    total_goal_conditions,
                                    completed_goal_conditions/total_goal_conditions))


    def save_results(self):
        if args.mask_type == 'true':
            mask_str = 'true_mask'
        elif args.detector_type == 'all':
            mask_str = 'pred_all_mask'
        elif args.detector_type == 'sep':
            mask_str = 'pred_sep_mask'

        successes, failures = self.successes, self.failures
        num_eval = len(successes) + len(failures)
        # overall results
        completed_goal_conditions = sum([entry['completed_goal_conditions'] for entry in successes]) + \
                                   sum([entry['completed_goal_conditions'] for entry in failures])
        total_goal_conditions = sum([entry['total_goal_conditions'] for entry in successes]) + \
                               sum([entry['total_goal_conditions'] for entry in failures])

        res = {
            'split': args.eval_split,
            'mask': mask_str,
            "SR":  "%d/%d = %.3f"%(len(successes),num_eval, len(successes)/num_eval),
            "GC": " %d/%d = %.3f"%(completed_goal_conditions,
                                    total_goal_conditions,
                                    completed_goal_conditions/total_goal_conditions),
        }
        save_path =args.save_name.replace('.log', '.json')
        with open(save_path, 'w') as r:
            json.dump(res, r, indent=2)


if __name__ == '__main__':

    # multiprocessing settings
    mp.set_start_method('spawn')
    manager = mp.Manager()

    # parser
    parser = argparse.ArgumentParser()
    Config(parser)

    # settings
    parser.add_argument('--reward_config', default='models/config/rewards.json')
    parser.add_argument('--smooth_nav', dest='smooth_nav', action='store_true', help='smooth nav actions (might be required based on training data)')
    parser.add_argument('--mask_type', default='true')
    parser.add_argument('--eval_split', type=str, default='valid_seen', choices=['train', 'valid_seen', 'valid_unseen'])

    # parse arguments
    args = parser.parse_args()
    args.save_path = os.path.join(args.pp_data, 'eval_with_repeat')
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    args.save_name = os.path.join(args.save_path, 'replaycheck_'+args.eval_split + '_' +args.mask_type+'.log')
    if os.path.exists(args.save_name):
        os.remove(args.save_name)

    # eval mode
    eval = EvalExpertTrajs(args, manager)

    # start threads
    eval.spawn_threads()