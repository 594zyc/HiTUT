import os, time, sys, logging, copy, shutil, glob
import json
import cv2
import numpy as np
from PIL import Image
from datetime import datetime

import torch

sys.path.append(os.path.join(os.environ['ALFRED_ROOT']))

from gen.constants import *
from gen.utils.image_util import decompress_mask
from gen.utils.video_util import VideoSaver, save_frame_failure
from env.thor_env import ThorEnv
from models.eval.eval import Eval, EvalMMT


class EvalTaskMMT(EvalMMT):
    """iTHOR-based interactive evaluation

    Based on eval_task.py form the original Alfred repository.
    Modified to fit the proposed MMT model
    """
    def start(self):
        for k in ['avg_fps', 'gc_completed', 'gc_total', 'plw_total', 'plw_s_spl', 'plw_gc_spl']:
            self.stats[k] = 0
        self.results['success'] = []
        self.results['failure'] = []
        self.spawn_threads()
        self.save_results()

    def save_results(self):
        per_task_sr = {}
        for k in self.stats:
            if '_success' in k:
                s_num = self.stats[k]
                t_num = self.stats[k.replace('success', 'total')]
                per_task_sr[k[:-8]] = "%d/%d = %.4f" %(s_num, t_num, s_num/t_num)
        save_dir = self.args.log_dir.replace('.log', '')
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        with open(os.path.join(save_dir, 'results.json'), 'w') as f:
            json.dump(self.results.copy(), f, indent=2)
        with open(os.path.join(save_dir, 'stats.json'), 'w') as f:
            json.dump(self.stats.copy(), f, indent=2)
        with open(os.path.join(save_dir, 'per_task_sr.json'), 'w') as f:
            json.dump(per_task_sr, f, indent=2)


    @classmethod
    def run(cls, n, model, detector, vocabs, task_queue, args, lock, stats, results):
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
            try:
                with open(os.path.join(task_path, 'ann_%d.json'%r_idx), 'r') as f:
                    traj_pp = json.load(f)
                with open(os.path.join(traj_pp['raw_path'], 'traj_data.json'), 'r') as f:
                    traj_raw = json.load(f)
                traj_pp['raw'] =traj_raw
                cls.evaluate(env, model, detector, vocabs, traj_pp, args, lock,
                    stats, results)
            except Exception as e:
                import traceback
                traceback.print_exc()
                logging.info("Error: " + repr(e))
                # quit()

        cv2.destroyAllWindows()
        # stop THOR
        env.stop()

    @classmethod
    def evaluate(cls, env, model, detector, vocabs, traj_data, args, lock, stats, results):

        prtfunc = logging.debug
        prtfunc('-'*50)
        prtfunc(traj_data['pp_path'])

        if isinstance(model, dict):
            model_navi = model['navi']
            model_mani = model['mani']
        else:
            model_navi = model_mani = model

        # setup scene
        cls.setup_scene(env, traj_data['raw'], args, reward_type='dense')
        try:
            horizon = traj_data['raw']['scene']['init_action']['horizon']
            rotation = traj_data['raw']['scene']['init_action']['rotation']
        except:
            horizon = rotation = 0

        all_done, success = False, False
        high_idx, high_fails, high_steps, api_fails = 0, 0, 0, 0
        high_history = ['[SOS]', 'None']
        high_history_before_last_navi = copy.deepcopy(high_history)
        low_history_col = []
        reward = 0
        t = 0
        stop_cond = ''
        failed_events= ''
        terminate = False

        path_name = traj_data['raw']['task_id'] + '_ridx' + str(traj_data['repeat_idx'])
        visualize_info = {'save_path': os.path.join(args.log_dir.replace('.log', ''), path_name)}
        if args.save_video:
            if os.path.exists(visualize_info['save_path']):
                shutil.rmtree(visualize_info['save_path'])
            os.makedirs(visualize_info['save_path'])

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

            if args.use_gt_high_action:
                gt_high_action = vocabs['out_vocab_high_type'].id2w(traj_data['high']['dec_out_high_actions'][high_idx])
                gt_high_arg = vocabs['out_vocab_arg'].id2w(traj_data['high']['dec_out_high_args'][high_idx])
                high_action = gt_high_action
                high_arg = gt_high_arg
            else:
                with torch.no_grad():
                    curr_frame = env.last_event.frame
                    masks, boxes, classes, scores = detector.get_preds_step(curr_frame)
                    # prtfunc('MaskRCNN Top8: ' + ''.join(['(%s, %.3f)'%(i, j) for i, j in zip(classes[:8], scores[:8])]))

                    observations = {}
                    observations['lang'] = traj_data['lang']['goal'] if not args.eval_disable_feat_lang_high else None
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
                visualize_info['high_idx'] = high_idx
                visualize_info['high_action'] = high_action
                visualize_info['high_arg'] = high_arg

                prtfunc('-'*50)
                prtfunc('Task goal: ' + ''.join(traj_data['lang']['goal_tokenize']).replace('  ', ' '))
                prtfunc('High proposals:')
                prtfunc('action: ' + ''.join(['(%s, %.3f)'%(high_abbr[i], j) for i, j in zip(high_actions, probs['type'])]))
                prtfunc('arg: ' + ''.join(['(%s, %.3f)'%(i, j) for i, j in zip(high_args, probs['arg'])]))

            if high_action == 'GotoLocation':
                high_history_before_last_navi = copy.deepcopy(high_history)


            if high_action == 'NoOp':
                all_done = True
                stop_cond += 'Predictes a high-level NoOp to terminate!'
                break

            if args.use_gt_navigation and high_action == 'GotoLocation':
                try:
                    gt_high_arg = vocabs['out_vocab_arg'].id2w(traj_data['high']['dec_out_high_args'][high_idx])
                    high_arg = gt_high_arg
                    prtfunc('Use ground truth navigation actions')
                except:
                    pass

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

                if args.use_gt_navigation and high_action == 'GotoLocation':
                    try:
                        proposals = [(vocabs['out_vocab_low_type'].id2w(traj_data['low']['dec_out_low_actions'][high_idx][low_idx]), None, None)]
                        prtfunc('Use ground truth navigation actions')
                    except:
                        stop_cond += 'Terminate due to do not find proper ground truth navigation actions! '
                        all_done = True
                        break

                else:
                    with torch.no_grad():
                        task_type = 'low_navi' if high_action == 'GotoLocation' else 'low_mani'
                        model = model_navi if task_type == 'low_navi' else model_mani
                        # visual observation
                        curr_frame = env.last_event.frame
                        masks, boxes, classes, scores = detector.get_preds_step(curr_frame)
                        prtfunc('MaskRCNN Top8: ' + ''.join(['(%s, %.3f)'%(i, j) for i, j in zip(classes[:8], scores[:8])]))

                        # disable language directives when retry the navigation subgoal
                        global_disable = args.eval_disable_feat_lang_navi if 'navi' in task_type else args.eval_disable_feat_lang_mani
                        use_lang = not global_disable and high_idx == len(low_history_col) and high_idx<len(traj_data['lang']['instr'])
                        if use_lang:
                            prtfunc('Instruction: ' + ''.join(traj_data['lang']['instr_tokenize'][high_idx]).replace('  ', ' '))
                        use_vis_his = args.eval_enable_feat_vis_his and 'navi' in task_type
                        use_pos = args.eval_enable_feat_posture and 'navi' in task_type


                        observations = {}
                        observations['lang'] = traj_data['lang']['instr'][high_idx] if use_lang else None
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
                    elif args.use_gt_mask:
                        mask = decompress_mask(traj_data['low']['mask'][high_idx][low_idx])
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

                    if args.save_video:
                        visualize_info['low_idx'] = low_idx
                        visualize_info['low_action'] = action
                        visualize_info['low_arg'] = low_arg
                        visualize_info['global_step'] = t
                        visualize_info['mask'] = mask
                        visualize_info['bbox'] = boxes[mask_id - 1] if mask is not None else None
                        visualize_info['class'] = mask_cls_pred  if mask is not None else None
                        visualize_info_ = visualize_info
                    else:
                        visualize_info_ = None

                    if action not in NON_INTERACT_ACTIONS and mask is None:
                        if args.save_video:
                            save_frame_failure(env.last_event.frame[:, :, ::-1], visualize_info_)
                        prev_t_success = False
                        continue

                    # use action and predicted mask (if available) to interact with the env
                    t_success, _, _, err, _ = env.va_interact(action, interact_mask=mask,
                        smooth_nav=args.smooth_nav, debug=args.debug, visualize_info=visualize_info_)
                    t += 1
                    prev_t_success = t_success

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
                        t_reward, t_done = env.get_transition_reward()
                        reward += t_reward
                        low_idx += 1
                        break
                    else:
                        api_fails += 1
                        if api_fails >= args.max_fails:
                            stop_cond = 'Reach 10 Api fails'
                            terminate = True
                        failed_events += 'bad action: %s(%s) api fail: %d |'%(action, low_arg, api_fails)
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

        if args.save_video:
            img = np.array(env.last_event.frame[:, :, ::-1])
            cv2.putText(img, 'Completed!', (5,20), cv2.FONT_HERSHEY_SIMPLEX ,0.5, (0,0,255), 1)
            a_disp = 'A%d: End'%(t+1)
            cv2.putText(img, a_disp, (5,40), cv2.FONT_HERSHEY_SIMPLEX ,0.5, (0,0,255), 1)
            for i in range(10):
                im_ind = len(glob.glob(visualize_info['save_path'] + '/*.png'))
                cv2.imwrite(visualize_info['save_path'] + '/%05d.png' %im_ind, img)
            images_path = visualize_info['save_path'] + '/*.png'
            video_path = visualize_info['save_path'] + '_%s.mp4'%traj_data['raw']['task_type']
            video_saver = VideoSaver()
            video_saver.save(images_path, video_path)

        # check if goal was satisfied
        goal_satisfied = env.get_goal_satisfied()
        if goal_satisfied:
            success = True

        # goal_conditions
        pcs = env.get_goal_conditions_met()
        goal_condition_success_rate = pcs[0] / float(pcs[1])

        # SPL
        path_len_weight = len(traj_data['raw']['plan']['low_actions'])
        factor = min(1., path_len_weight / float(t))
        s_spl = (1 if goal_satisfied else 0) *factor
        pc_spl = goal_condition_success_rate * factor

        # path length weighted SPL
        plw_s_spl = s_spl * path_len_weight
        plw_pc_spl = pc_spl * path_len_weight

        # log success/fails
        lock.acquire()

        logging.info("-------------"*5)
        logging.info("Task ID: %s (ridx: %d)" % (traj_data['raw']['task_id'], traj_data['repeat_idx']))
        logging.info("Task type: %s" % (traj_data['raw']['task_type']))
        total_time = time.time() - t_start_total
        fps = t / total_time
        logging.info('Time cost: %.1f sec, FPS: %.2f'%(total_time, fps))

        gt_high_type = vocabs['out_vocab_high_type'].seq_decode(traj_data['high']['dec_out_high_actions'])
        gt_high_arg = vocabs['out_vocab_arg'].seq_decode(traj_data['high']['dec_out_high_args'])
        for hidx in range(len(traj_data['lang']['instr_tokenize'])):
            # try:
            logging.info('[%d] Instr: '%(hidx+1) + ''.join(traj_data['lang']['instr_tokenize'][hidx]).replace('  ', ' '))
            logging.info('Gold High: %s(%s)'%(high_abbr[gt_high_type[hidx]], gt_high_arg[hidx]))
            gt_low_type = vocabs['out_vocab_low_type'].seq_decode(traj_data['low']['dec_out_low_actions'][hidx])
            if gt_high_type[hidx] == 'GotoLocation':
                logging.info('Gold Low: '+' '.join(['%s'%(navi_abbr[i]) for i in gt_low_type]))
            else:
                gt_low_arg = vocabs['out_vocab_arg'].seq_decode(traj_data['low']['dec_out_low_args'][hidx])
                logging.info('Gold Low: '+' '.join(['%s %s'%(i, gt_low_arg[idx]) for idx,i in enumerate(gt_low_type)]))

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

        logging.info('Task goal: ' + ''.join(traj_data['lang']['goal_tokenize']).replace('  ', ' '))
        logging.info('High gold: '+' '.join(['%s(%s)'%(high_abbr[gt_high_type[i]], gt_high_arg[i]) for i in range(len(gt_high_type))]))
        logging.info('High pred: '+' '.join(['%s(%s)'%(high_abbr[i], high_history[2*idx+3]) for idx, i in enumerate(high_history[2:][::2])]))
        logging.info('Stop condition: %s'%stop_cond)
        logging.info('Fail num high %d | api: %d'%(high_fails, api_fails))
        logging.info('Path long expert/model: %d / %d = %.3f'%(path_len_weight, t, factor))
        logging.info("Task succeed: %s"%str(success))
        logging.info("Goal conditions meet: %d / %d"%(pcs[0], pcs[1]))

        stats['gc_completed'] += int(pcs[0])
        stats['gc_total'] += int(pcs[1])
        stats['plw_total'] += path_len_weight
        stats['plw_s_spl'] += float(plw_s_spl)
        stats['plw_gc_spl'] += float(plw_pc_spl)
        task_type = traj_data['raw']['task_type']
        if task_type+'_total' not in stats:
            stats[task_type + '_total'] = 0
            stats[task_type + '_success'] = 0
        stats[task_type + '_total'] += 1
        stats[task_type + '_success'] += int(success)

        log_entry = {'trial': traj_data['raw']['task_id'],
                     'type': traj_data['raw']['task_type'],
                     'repeat_idx': traj_data['repeat_idx'],
                     'reward': float(reward)}
        if success:
            results['success'] += [log_entry]
        else:
            results['failure'] += [log_entry]
        num_success = len(results['success'])
        num_failure = len(results['failure'])
        num_eval = num_success+ num_failure
        avg_fps_new = (stats['avg_fps'] * (num_eval - 1) + fps) / num_eval
        stats['avg_fps'] = avg_fps_new

        num_gc_completed = stats['gc_completed']
        num_gc_total = stats['gc_total']

        sr = 100*num_success / num_eval
        tsr = 100*stats[task_type + '_success'] / stats[task_type + '_total']
        pc =  100*num_gc_completed / num_gc_total
        plw_sr = 100*stats['plw_s_spl'] / stats['plw_total']
        plw_pc = 100*stats['plw_gc_spl'] / stats['plw_total']

        # prtfunc = print if num_eval != stats['task_num'] else logging.info
        logging.info('Eval result so far: ')
        elapsed = (time.time()- stats['start_time'])/60
        logging.info('%d/%d evaluated | elapsed: %.1f min | avg fps: %.2f (x %d procs)'%(num_eval,
            stats['task_num'], elapsed, avg_fps_new, args.num_threads))
        logging.info("SR: %d/%d = %.2f (PW: %.2f)" % (num_success, num_eval, sr, plw_sr))
        logging.info("GC: %d/%d = %.2f (PW: %.2f)" % (num_gc_completed, num_gc_total, pc, plw_pc))
        logging.info("Task: %s SR: %d/%d = %.2f" % (task_type, stats[task_type + '_success'], stats[task_type + '_total'], tsr))
        lock.release()


class EvalTask(Eval):
    '''
    evaluate overall task performance
    '''

    @classmethod
    def run(cls, model, resnet, task_queue, args, lock, successes, failures, results):
        '''
        evaluation loop
        '''
        # start THOR
        env = ThorEnv()

        while True:
            if task_queue.qsize() == 0:
                break

            task = task_queue.get()

            try:
                traj = model.load_task_json(task)
                r_idx = task['repeat_idx']
                print("Evaluating: %s" % (traj['root']))
                print("No. of trajectories left: %d" % (task_queue.qsize()))
                cls.evaluate(env, model, r_idx, resnet, traj, args, lock, successes, failures, results)
            except Exception as e:
                import traceback
                traceback.print_exc()
                print("Error: " + repr(e))

        # stop THOR
        env.stop()


    @classmethod
    def evaluate(cls, env, model, r_idx, resnet, traj_data, args, lock, successes, failures, results):
        # reset model
        model.reset()

        # setup scene
        reward_type = 'dense'
        cls.setup_scene(env, traj_data, r_idx, args, reward_type=reward_type)

        # mask only evaluation
        expert_init_actions_all = traj_data['plan']['low_actions']

        # extract language features
        feat = model.featurize([traj_data], load_mask=False)

        # goal instr
        goal_instr = traj_data['turk_annotations']['anns'][r_idx]['task_desc']

        done, success = False, False
        fails = 0
        t = 0
        reward = 0

        t_start_total = time.time()

        while not done:
            # break if max_steps reached
            if t >= args.max_steps:
                break

            # extract visual features
            curr_image = Image.fromarray(np.uint8(env.last_event.frame))
            feat['frames'] = resnet.featurize([curr_image], batch=1).unsqueeze(0)

            # forward model
            m_out = model.step(feat)
            m_pred = model.extract_preds(m_out, [traj_data], feat, clean_special_tokens=False)
            m_pred = list(m_pred.values())[0]

            # check if <<stop>> was predicted
            if m_pred['action_low'] == cls.STOP_TOKEN:
                print("\tpredicted STOP")
                break

            # get action and mask
            action, mask = m_pred['action_low'], m_pred['action_low_mask'][0]
            if args.only_eval_mask:
                try:
                    action = expert_init_actions_all[t]['discrete_action']['action']
                except:
                    action = "<<stop>>"
            mask = np.squeeze(mask, axis=0) if model.has_interaction(action) else None


            # print action
            if args.debug:
                print('pred:', action)

            # use predicted action and mask (if available) to interact with the env
            t_success, _, _, err, _ = env.va_interact(action, interact_mask=mask, smooth_nav=args.smooth_nav, debug=args.debug)
            if not t_success:
                fails += 1
                if fails >= args.max_fails:
                    print("Interact API failed %d times" % fails + "; latest error '%s'" % err)
                    break

            # next time-step
            t_reward, t_done = env.get_transition_reward()
            reward += t_reward
            t += 1

        total_time = time.time() - t_start_total
        print('total time', total_time, t / total_time, 'fps')


        # check if goal was satisfied
        goal_satisfied = env.get_goal_satisfied()
        if goal_satisfied:
            print("Goal Reached")
            success = True


        # goal_conditions
        pcs = env.get_goal_conditions_met()
        goal_condition_success_rate = pcs[0] / float(pcs[1])

        # SPL
        path_len_weight = len(traj_data['plan']['low_actions'])
        s_spl = (1 if goal_satisfied else 0) * min(1., path_len_weight / float(t))
        pc_spl = goal_condition_success_rate * min(1., path_len_weight / float(t))

        # path length weighted SPL
        plw_s_spl = s_spl * path_len_weight
        plw_pc_spl = pc_spl * path_len_weight

        # log success/fails
        lock.acquire()
        log_entry = {'trial': traj_data['task_id'],
                     'type': traj_data['task_type'],
                     'repeat_idx': int(r_idx),
                     'goal_instr': goal_instr,
                     'completed_goal_conditions': int(pcs[0]),
                     'total_goal_conditions': int(pcs[1]),
                     'goal_condition_success': float(goal_condition_success_rate),
                     'success_spl': float(s_spl),
                     'path_len_weighted_success_spl': float(plw_s_spl),
                     'goal_condition_spl': float(pc_spl),
                     'path_len_weighted_goal_condition_spl': float(plw_pc_spl),
                     'path_len_weight': int(path_len_weight),
                     'reward': float(reward)}
        if success:
            successes.append(log_entry)
        else:
            failures.append(log_entry)

        # overall results
        results['all'] = cls.get_metrics(successes, failures)

        print("-------------")
        print("SR: %d/%d = %.3f" % (results['all']['success']['num_successes'],
                                    results['all']['success']['num_evals'],
                                    results['all']['success']['success_rate']))
        print("GC: %d/%d = %.3f" % (results['all']['goal_condition_success']['completed_goal_conditions'],
                                    results['all']['goal_condition_success']['total_goal_conditions'],
                                    results['all']['goal_condition_success']['goal_condition_success_rate']))
        print("PLW SR: %.3f" % (results['all']['path_length_weighted_success_rate']))
        print("PLW GC: %.3f" % (results['all']['path_length_weighted_goal_condition_success_rate']))
        print("-------------")

        # task type specific results
        task_types = ['pick_and_place_simple', 'pick_clean_then_place_in_recep', 'pick_heat_then_place_in_recep',
                      'pick_cool_then_place_in_recep', 'pick_two_obj_and_place', 'look_at_obj_in_light',
                      'pick_and_place_with_movable_recep']
        for task_type in task_types:
            task_successes = [s for s in (list(successes)) if s['type'] == task_type]
            task_failures = [f for f in (list(failures)) if f['type'] == task_type]
            if len(task_successes) > 0 or len(task_failures) > 0:
                results[task_type] = cls.get_metrics(task_successes, task_failures)
            else:
                results[task_type] = {}

        lock.release()

    @classmethod
    def get_metrics(cls, successes, failures):
        '''
        compute overall succcess and goal_condition success rates along with path-weighted metrics
        '''
        # stats
        num_successes, num_failures = len(successes), len(failures)
        num_evals = len(successes) + len(failures)
        total_path_len_weight = sum([entry['path_len_weight'] for entry in successes]) + \
                                sum([entry['path_len_weight'] for entry in failures])
        completed_goal_conditions = sum([entry['completed_goal_conditions'] for entry in successes]) + \
                                   sum([entry['completed_goal_conditions'] for entry in failures])
        total_goal_conditions = sum([entry['total_goal_conditions'] for entry in successes]) + \
                               sum([entry['total_goal_conditions'] for entry in failures])

        # metrics
        sr = float(num_successes) / num_evals
        pc = completed_goal_conditions / float(total_goal_conditions)
        plw_sr = (float(sum([entry['path_len_weighted_success_spl'] for entry in successes]) +
                        sum([entry['path_len_weighted_success_spl'] for entry in failures])) /
                  total_path_len_weight)
        plw_pc = (float(sum([entry['path_len_weighted_goal_condition_spl'] for entry in successes]) +
                        sum([entry['path_len_weighted_goal_condition_spl'] for entry in failures])) /
                  total_path_len_weight)

        # result table
        res = dict()
        res['success'] = {'num_successes': num_successes,
                          'num_evals': num_evals,
                          'success_rate': sr}
        res['goal_condition_success'] = {'completed_goal_conditions': completed_goal_conditions,
                                        'total_goal_conditions': total_goal_conditions,
                                        'goal_condition_success_rate': pc}
        res['path_length_weighted_success_rate'] = plw_sr
        res['path_length_weighted_goal_condition_success_rate'] = plw_pc

        return res

    def create_stats(self):
            '''
            storage for success, failure, and results info
            '''
            self.successes, self.failures = self.manager.list(), self.manager.list()
            self.results = self.manager.dict()

    def save_results(self):
        results = {'successes': list(self.successes),
                   'failures': list(self.failures),
                   'results': dict(self.results)}

        save_path = os.path.dirname(self.args.model_path)
        save_path = os.path.join(save_path, 'task_results_' + self.args.eval_split + '_' + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + '.json')
        with open(save_path, 'w') as r:
            json.dump(results, r, indent=4, sort_keys=True)

