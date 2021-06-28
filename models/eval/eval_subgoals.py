import os, sys, json, logging, time, re

import numpy as np
from PIL import Image
import torch
from datetime import datetime

sys.path.append(os.path.join(os.environ['ALFRED_ROOT']))

from gen.constants import *
from gen.utils.image_util import decompress_mask
from env.thor_env import ThorEnv
from models.eval.eval import Eval, EvalMMT


class EvalSubgoalsMMT(EvalMMT):
    '''
    evaluate subgoals by teacher-forching expert demonstrations
    '''


    def start(self):
        # subgoal types
        ALL_SUBGOALS = ['GotoLocation', 'PickupObject', 'PutObject', 'CoolObject', 'HeatObject',
                                       'CleanObject', 'SliceObject', 'ToggleObject']

        # make subgoals list
        sg_to_eval = ALL_SUBGOALS if self.args.subgoals.lower() == "all" else self.args.subgoals.split(',')
        sg_to_eval = [sg for sg in sg_to_eval if sg in ALL_SUBGOALS]
        if not sg_to_eval:
            logging.info("No valid subgoal to be evaluated. ")
            quit()
        self.stats['sg_to_eval'] = sg_to_eval
        logging.info("Subgoals to evaluate: %s" % str(sg_to_eval))
        if self.args.use_gt_navigation:
            logging.info("Using ground truth navigation actions makes no sense in subgoal evaluation!")
            quit()

        # create results collections
        self.stats['success_ori'] = {sg:0 for sg in sg_to_eval}
        self.stats['success_corr'] = {sg:0 for sg in sg_to_eval}
        self.stats['failure_ori'] = {sg:0 for sg in sg_to_eval}
        self.stats['failure_corr'] = {sg:0 for sg in sg_to_eval}
        self.stats['expert_pl'] = {sg:0 for sg in sg_to_eval}
        self.stats['s_spl_ori'] = {sg:0 for sg in sg_to_eval}
        self.stats['plw_s_spl_ori'] = {sg:0 for sg in sg_to_eval}
        self.stats['s_spl_corr'] = {sg:0 for sg in sg_to_eval}
        self.stats['plw_s_spl_corr'] = {sg:0 for sg in sg_to_eval}
        self.stats['task_complete'] = 0
        self.stats['fail_reason'] = {sg: dict() for sg in sg_to_eval}
        self.stats['navi_attempts'] = {}

        self.spawn_threads()
        EvalSubgoalsMMT.save_results(self.args, self.stats)

    # def queue_tasks(self):
    #     self.stats = self.manager.dict()
    #     self.results = self.manager.dict()

    #     # queue tasks
    #     self.task_queue = self.manager.Queue()


    #     with open(os.path.join(self.args.eval_path, 'ckpt_seen_subgoal_valid_unseen_maxfail9_haspos_/stats.json'), 'r') as f:
    #         self.stats.update(json.load(f))
    #     evaluted = {}
    #     with open(os.path.join(self.args.eval_path, 'ckpt_seen_subgoal_valid_unseen_maxfail9_haspos_.log'), 'r') as f:
    #         for line in f.readlines():
    #             if 'Task ID' in line:
    #                 task_id = re.search(r'Task ID: ([\w]+) \(ridx: (\d)\)', line).group(1)
    #                 ridx = re.search(r'Task ID: ([\w]+) \(ridx: (\d)\)', line).group(2)
    #                 evaluted[task_id + '_' + ridx] = 1

    #     self.task_queue = self.manager.Queue()
    #     for split in self.eval_splits:
    #         # random.shuffle(self.dataset.dataset_splits[split])
    #         for task in self.dataset.dataset_splits[split]:
    #             task_path = os.path.join(self.pp_path, split, task['task'])
    #             task_id = task_path.split('/')[-1]
    #             if (task_id+'_'+str(task['repeat_idx'])) in evaluted:
    #                 continue
    #             self.task_queue.put((task_path, task['repeat_idx']))
    #             if self.args.fast_epoch and self.task_queue.qsize() == 3:
    #                 break
    #         else:
    #             continue
    #         break
    #     self.stats['task_num'] = self.task_queue.qsize()
    #     print('Total task num:', self.task_queue.qsize())


    @classmethod
    def save_results(cls, args, stats):
        stats_cp = stats.copy()
        for sg in stats['sg_to_eval']:
            fail_num = stats['failure_corr'][sg]
            stats_cp['fail_reason'][sg]['fail_num'] = fail_num
            for k,v in stats['fail_reason'][sg].items():
                stats_cp['fail_reason'][sg]['fail_prop_'+k] = 100 * v / (fail_num+10e-8)

        save_dir = args.log_dir.replace('.log', '')
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        # with open(os.path.join(save_dir, 'results.json'), 'w') as f:
        #     json.dump(self.results.copy(), f, indent=2)
        with open(os.path.join(save_dir, 'stats_1.json'), 'w') as f:
            json.dump(stats_cp, f, indent=2)


    @classmethod
    def run(cls, n, model, detector, vocabs, task_queue, args, lock, stats, results):
        '''
        evaluation loop
        '''
        num_core_per_proc = args.num_core_per_proc
        cores = ','.join([str(i + num_core_per_proc*n) for i in range(num_core_per_proc)])
        core_mask_op = "taskset -pc %s %d" %(cores, os.getpid())
        os.system(core_mask_op)
        print('start process: %d (pid: %d) op: %s'%(n, os.getpid(), core_mask_op))

        try:
            # start THOR
            env = ThorEnv()

            # set logger
            log_level = logging.DEBUG if args.debug else logging.INFO
            log_handlers = [logging.StreamHandler(), logging.FileHandler(args.log_dir)]
            logging.basicConfig(handlers=log_handlers, level=log_level,
                format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')


            while True:
                lock.acquire()
                if task_queue.qsize() == 0:
                    break
                task_path, r_idx = task_queue.get()
                lock.release()

                try:
                    with open(os.path.join(task_path, 'ann_%d.json'%r_idx), 'r') as f:
                        traj_pp = json.load(f)
                    with open(os.path.join(traj_pp['raw_path'], 'traj_data.json'), 'r') as f:
                        traj_raw = json.load(f)
                    traj_pp['raw'] =traj_raw
                    num_high = len(traj_pp['lang']['instr_tokenize'])
                    traj_subgoal_ids = traj_pp['high']['dec_out_high_actions']
                    for eval_idx in range(num_high):
                        sub_goal =  vocabs['out_vocab_high_type'].id2w(traj_subgoal_ids[eval_idx])
                        if sub_goal not in stats['sg_to_eval']:
                            continue
                        logging.debug('-'*50)
                        logging.debug('eval subgoal: ' + sub_goal)
                        cls.evaluate(eval_idx, env, model, detector, vocabs, traj_pp, args, lock, stats, results)
                    lock.acquire()
                    stats['task_complete'] += 1
                    lock.release()
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print("Error: " + repr(e))
                    quit()

            # stop THOR
            env.stop()
        except Exception as e:
            import traceback
            traceback.print_exc()
            logging.info("Error: " + repr(e))

    @classmethod
    def evaluate(cls, eval_idx, env, model, detector, vocabs, traj_data, args, lock, stats, results):

        prtfunc = logging.debug

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

        # get expert actions
        gt_low_actions, gt_masks, interacts = [], [], []
        for hidx, low_action_seq in enumerate(traj_data['low']['dec_in_low_actions']):
            if hidx == eval_idx:
                break
            for low_idx, low_action_idx in enumerate(low_action_seq[1:-1]):
                gt_low_actions.append(vocabs['in_vocab_action'].id2w(low_action_idx))
                interact = traj_data['low']['interact'][hidx][low_idx]
                interacts.append(interact)
                if not interact:
                    gt_masks.append(None)
                else:
                    gt_masks.append(decompress_mask(traj_data['low']['mask'][hidx][low_idx]))

        # expert actions
        gt_high_types = vocabs['in_vocab_action'].seq_decode(traj_data['high']['dec_in_high_actions'])
        gt_high_args = vocabs['in_vocab_action'].seq_decode(traj_data['high']['dec_in_high_args'])
        gt_high_action = gt_high_types[eval_idx+1]
        gt_high_arg = gt_high_args[eval_idx+1]
        gt_low_type = vocabs['out_vocab_low_type'].seq_decode(traj_data['low']['dec_out_low_actions'][eval_idx])
        gt_low_arg = vocabs['out_vocab_arg'].seq_decode(traj_data['low']['dec_out_low_args'][eval_idx])

        # execute expert actions
        for step_idx in range(len(gt_low_actions)):
            # get action and mask
            action = gt_low_actions[step_idx]
            mask = gt_masks[step_idx]
            interact = interacts[step_idx]
            if mask is not None:
                mask = np.squeeze(mask)

            # use action and predicted mask (if available) to interact with the env
            t_success, _, _, err, _ = env.va_interact(action, interact_mask=mask, smooth_nav=args.smooth_nav, debug=args.debug)
            if not t_success:
                print("expert initialization failed")
                print(traj_data['pp_path'] + '\nstep idx:' + str(step_idx))
                return

            # next time-step
            _, _ = env.get_transition_reward()

        # prepare model input
        high_history = []
        for i in range(eval_idx+1):
            high_history += [gt_high_types[i], gt_high_args[i]]

        # high-level model prediction
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


        # only use the top1 proposal in subgoal evaluation
        high_action = high_actions[0]
        high_arg = high_args[0]


        prtfunc('-'*50)
        prtfunc('Task goal: ' + ''.join(traj_data['lang']['goal_tokenize']).replace('  ', ' '))
        prtfunc('High gold: '+' '.join(['%s(%s)'%(high_abbr[gt_high_types[i+1]], gt_high_args[i+1]) for i in range(len(gt_high_types)-1)]))
        prtfunc('Evaluate: %s(%s)   idx: %d'%(gt_high_action, gt_high_arg, eval_idx))
        prtfunc('High pred: %s(%s)'%(high_action, high_arg))


        # go into the low-level action prediction loop
        sg_succ_original, sg_succ_correct = False, False
        terminate = False
        t, reward, attempts = 0, 0, 0
        low_history_col = []
        stop_cond = ''
        while not terminate and attempts < args.max_high_fails:
            low_idx = 0
            low_history = [high_action, high_arg]
            low_vis_history = []
            subgoal_done = False
            while not subgoal_done:
                # break if max_steps reached
                if low_idx >= args.max_low_steps:
                    stop_cond = 'Exceed step limitation'
                    terminate = True
                    break

                prtfunc('-'*50)
                prtfunc('Completing subgoal: %s(%s)'%(high_action, high_arg))
                prtfunc('Instruction: ' + ''.join(traj_data['lang']['instr_tokenize'][eval_idx]).replace('  ', ' '))

                with torch.no_grad():
                    task_type = 'low_navi' if high_action == 'GotoLocation' else 'low_mani'
                    model = model_navi if task_type == 'low_navi' else model_mani

                    # visual observation
                    curr_frame = env.last_event.frame
                    masks, boxes, classes, scores = detector.get_preds_step(curr_frame)
                    prtfunc('MaskRCNN Top8: ' + ''.join(['(%s, %.3f)'%(i, j) for i, j in zip(classes[:8], scores[:8])]))

                    disable_lang = args.eval_disable_feat_lang_navi if 'navi' in task_type else args.eval_disable_feat_lang_mani
                    use_lang = not disable_lang
                    use_vis_his = args.eval_enable_feat_vis_his and 'navi' in task_type
                    use_pos = args.eval_enable_feat_posture and 'navi' in task_type

                    observations = {}
                    observations['lang'] = traj_data['lang']['instr'][eval_idx] if use_lang else None
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

                t_success = False
                for action, low_arg, mask_id in proposals:
                    if action == 'NoOp':
                        low_history += [action, low_arg]
                        low_vis_history += [observations['vis']] if args.eval_enable_feat_vis_his else []
                        subgoal_done = True
                        prtfunc("Subgoal '%s(%s)' is done! low steps: %d"%(high_action, high_arg, low_idx+1))
                        break

                    # disable masks/arguments for non-ineractive actions
                    if action in NON_INTERACT_ACTIONS:
                        low_arg, mask = 'None', None
                    else:
                        try:
                            mask = masks[mask_id - 1]
                            mask_cls_pred = classes[mask_id-1]
                            if not similar(mask_cls_pred, low_arg):
                                prtfunc('Agent: no correct mask grounding!')
                                mask = None
                        except:
                            # if low_arg in classes:
                            #     mask = masks[classes.index(low_arg)]
                            if mask_id == 0:
                                prtfunc('Agent: no available mask')
                                mask = None
                            else:
                                prtfunc('Invaild mask id: %s'%str(mask_id))
                                mask = None


                    if action not in NON_INTERACT_ACTIONS and mask is None:
                        stop_cond = 'bad mask'
                        t_success = False
                        continue

                    # use action and predicted mask (if available) to interact with the env
                    t_success, _, _, err, _ = env.va_interact(action, interact_mask=mask,
                        smooth_nav=args.smooth_nav, debug=args.debug)
                    t += 1

                    if t_success:
                        low_history += [action, low_arg]
                        low_vis_history += [observations['vis']] if args.eval_enable_feat_vis_his else []
                        rotation += 90 if action == 'RotateRight' else -90 if action == 'RotateLeft' else 0
                        horizon += 15 if action == 'LookUp' else -15 if action == 'LookDown' else 0
                        prtfunc('low pred: %s(%s)'%(action, low_arg))
                        prtfunc('Agent posture: rotation: %d horizon: %d'%(rotation, horizon))
                        prtfunc('high idx: %d  low idx: %d'%(eval_idx, low_idx))
                        prtfunc('Successfully executed!')
                        prtfunc('Low history: '+' '.join(['%s'%i for i in low_history[::2]]))
                        t_reward, t_done = env.get_transition_reward()
                        reward += t_reward
                        low_idx += 1
                        break
                    else:
                        prtfunc('low pred: %s(%s)'%(action, low_arg))
                        prtfunc('Failed!')

                # update subgoals
                curr_subgoal_idx = env.get_subgoal_idx()
                if curr_subgoal_idx == eval_idx:
                    sg_succ_original = True
                    # break
                    # ^^^^
                    # As in the original evaluation setting of ALFRED, the subgoal evaluation will
                    # terminate here and term the process as success. But the model is not necessarily
                    # stop at the state (i.e. it does not know it has met the sub-goal) and may continue
                    # moving, which would cause a mismatch between subgoal and task evaluation.

                    # In our evaluation, alternatively, we do not terminate here but only let the agent to
                    # predict when to terminate by itself

                if subgoal_done:
                    attempts += 1
                    if curr_subgoal_idx == eval_idx:
                        stop_cond = 'Predicted a NoOp'
                        sg_succ_correct = True
                        terminate = True
                    elif high_action != 'GotoLocation':
                        stop_cond = 'Predicted a NoOp'
                        terminate = True
                    else:
                        low_history_col.extend(low_history[2:])
                    break

                elif not t_success:
                    attempts += 1
                    if high_action == 'GotoLocation':
                        low_history_col.extend(low_history[2:])
                    else:
                        if stop_cond == '':
                            stop_cond = 'No valid proposal'
                        if curr_subgoal_idx == eval_idx:
                            sg_succ_correct = True
                        terminate = True
                    break



        prtfunc('Stop condition: '+ stop_cond)
        prtfunc('Attempts: %d'%attempts)



        # metrics
        pl = len(low_history)/2
        expert_pl = len(gt_low_type)

        s_spl_ori = (1 if sg_succ_original else 0) * min(1., expert_pl / (pl + 1e-10))
        plw_s_spl_ori = s_spl_ori * expert_pl

        s_spl_corr = (1 if sg_succ_correct else 0) * min(1., expert_pl / (pl + 1e-10))
        plw_s_spl_corr = s_spl_corr * expert_pl

        # log success/fails
        lock.acquire()

        logging.info('-'*50)
        logging.info("Task ID: %s (ridx: %d)" % (traj_data['raw']['task_id'], traj_data['repeat_idx']))
        logging.info("Task type: %s" % (traj_data['raw']['task_type']))


        logging.info('Task goal: ' + ''.join(traj_data['lang']['goal_tokenize']).replace('  ', ' '))
        logging.info('Evaluate: %s(%s)   idx: %d'%(gt_high_action, gt_high_arg, eval_idx))
        logging.info('High pred: %s(%s)'%(high_action, high_arg))
        logging.info('Subgoal success: <ori> %s  <corr> %s'%(sg_succ_original, sg_succ_correct))


        logging.info('[Subgoal %d] Instr: '%(eval_idx+1) + ''.join(traj_data['lang']['instr_tokenize'][eval_idx]).replace('  ', ' '))

        if gt_high_action == 'GotoLocation':
            logging.info('Gold Low: '+' '.join(['%s'%(navi_abbr[i]) for i in gt_low_type]))
        else:
            logging.info('Gold Low: '+' '.join(['%s %s'%(i, gt_low_arg[idx]) for idx,i in enumerate(gt_low_type)]))
        try:
            if low_history[2:][0] in navi_abbr:
                logging.info('Prew Low: '+' '.join(['%s'%(navi_abbr[i]) for i in low_history_col[2:][::2]]))
            else:
                logging.info('Pred Low: '+' '.join(['%s'%(i) for i in low_history[2:]]))
        except:
            logging.info('Predition recording failed.')
        logging.info('Stop condition: ' + stop_cond)
        logging.info('Attempts: %d'%attempts)

        sg = gt_high_action

        temp = stats['expert_pl']
        temp[sg] += expert_pl
        stats['expert_pl'] = temp

        temp = stats['s_spl_ori']
        temp[sg] += s_spl_ori
        stats['s_spl_ori'] = temp

        temp = stats['plw_s_spl_ori']
        temp[sg] += plw_s_spl_ori
        stats['plw_s_spl_ori'] = temp

        temp = stats['s_spl_corr']
        temp[sg] += s_spl_corr
        stats['s_spl_corr'] = temp

        temp = stats['plw_s_spl_corr']
        temp[sg] += plw_s_spl_corr
        stats['plw_s_spl_corr'] = temp


        if not sg_succ_correct:
            temp = stats['fail_reason']
            if high_action == 'GotoLocation':
                fail_reason = 'Unable to navigate to the target in all attempts'
            elif stop_cond == 'Predicted a NoOp':
                try:
                    for idx, atype in enumerate(gt_low_type):
                        if low_history[idx*2] != atype:
                            fail_reason = 'incorrect type'
                            break
                        if low_history[idx*2+1] != gt_low_arg[idx]:
                            fail_reason = 'incorrect arg'
                            break
                except:
                    fail_reason = 'miss an action'
            else:
                fail_reason = stop_cond
            if fail_reason in temp[sg]:
                temp[sg][fail_reason] += 1
            else:
                temp[sg][fail_reason] = 1
            stats['fail_reason'] = temp
            logging.info('Fail reason: ' + fail_reason)

        if sg_succ_correct and sg == 'GotoLocation':
            temp = stats['navi_attempts']
            if attempts in temp:
                temp[attempts] += 1
            else:
                temp[attempts] = 1
            stats['navi_attempts'] = temp

        if sg_succ_original:
            succ = stats['success_ori']
            succ[sg]+=1
            stats['success_ori'] = succ
        else:
            fail = stats['failure_ori']
            fail[sg]+=1
            stats['failure_ori'] = fail
        if sg_succ_correct:
            succ = stats['success_corr']
            succ[sg]+=1
            stats['success_corr'] = succ
        else:
            fail = stats['failure_corr']
            fail[sg]+=1
            stats['failure_corr'] = fail

        # save results
        logging.info('-'*50)
        for m in ['ori', 'corr']:
            logging.info('Evaluation Mode: <%s>'%m)
            for sg in stats['sg_to_eval']:
                num_successes, num_failures = stats['success_'+m][sg], stats['failure_'+m][sg]
                num_evals = num_successes + num_failures
                if num_evals > 0:
                    sr = 100 * num_successes / num_evals
                    total_path_len_weight = stats['expert_pl'][sg]
                    sr_plw = 100 * stats['plw_s_spl_'+m][sg] / total_path_len_weight
                    logging.info("[%s] SR: %d/%d = %.2f (PW: %.2f)" % (sg, num_successes, num_evals, sr, sr_plw))
        elapsed = (time.time()- stats['start_time'])/60
        logging.info('Evaluated: %d / %d (elapsed: %.1f min)'%(stats['task_complete']+1, stats['task_num'], elapsed))

        if (stats['task_complete']+1) % 30 == 0:
            cls.save_results(args, stats)

        lock.release()


class EvalSubgoals(Eval):
    '''
    evaluate subgoals by teacher-forching expert demonstrations
    '''

    # subgoal types
    ALL_SUBGOALS = ['GotoLocation', 'PickupObject', 'PutObject', 'CoolObject', 'HeatObject', 'CleanObject', 'SliceObject', 'ToggleObject']

    @classmethod
    def run(cls, model, resnet, task_queue, args, lock, successes, failures, results):
        '''
        evaluation loop
        '''
        # start THOR
        env = ThorEnv()

        # make subgoals list
        subgoals_to_evaluate = cls.ALL_SUBGOALS if args.subgoals.lower() == "all" else args.subgoals.split(',')
        subgoals_to_evaluate = [sg for sg in subgoals_to_evaluate if sg in cls.ALL_SUBGOALS]
        print ("Subgoals to evaluate: %s" % str(subgoals_to_evaluate))

        # create empty stats per subgoal
        for sg in subgoals_to_evaluate:
            successes[sg] = list()
            failures[sg] = list()

        while True:
            if task_queue.qsize() == 0:
                break

            task = task_queue.get()

            try:
                traj = model.load_task_json(task)
                r_idx = task['repeat_idx']
                subgoal_idxs = [sg['high_idx'] for sg in traj['plan']['high_pddl'] if sg['discrete_action']['action'] in subgoals_to_evaluate]
                for eval_idx in subgoal_idxs:
                    print("No. of trajectories left: %d" % (task_queue.qsize()))
                    cls.evaluate(env, model, eval_idx, r_idx, resnet, traj, args, lock, successes, failures, results)
            except Exception as e:
                import traceback
                traceback.print_exc()
                print("Error: " + repr(e))

        # stop THOR
        env.stop()

    @classmethod
    def evaluate(cls, env, model, eval_idx, r_idx, resnet, traj_data, args, lock, successes, failures, results):
        # reset model
        model.reset()

        # setup scene
        reward_type = 'dense'
        cls.setup_scene(env, traj_data, r_idx, args, reward_type=reward_type)

        # expert demonstration to reach eval_idx-1
        expert_init_actions = [a['discrete_action'] for a in traj_data['plan']['low_actions'] if a['high_idx'] < eval_idx]
        expert_init_actions_all = traj_data['plan']['low_actions']


        # subgoal info
        subgoal_action = traj_data['plan']['high_pddl'][eval_idx]['discrete_action']['action']
        subgoal_instr = traj_data['turk_annotations']['anns'][r_idx]['high_descs'][eval_idx]

        # print subgoal info
        print("Evaluating: %s\nSubgoal %s (%d)\nInstr: %s" % (traj_data['root'], subgoal_action, eval_idx, subgoal_instr))

        # extract language features
        feat = model.featurize([traj_data], load_mask=False)

        # previous action for teacher-forcing during expert execution (None is used for initialization)
        prev_action = None

        done, subgoal_success = False, False
        fails = 0
        t = 0
        reward = 0
        while not done:
            # break if max_steps reached
            if t >= args.max_steps + len(expert_init_actions):
                break

            # extract visual feats
            curr_image = Image.fromarray(np.uint8(env.last_event.frame))
            feat['frames'] = resnet.featurize([curr_image], batch=1).unsqueeze(0)

            # expert teacher-forcing upto subgoal
            if t < len(expert_init_actions):
                # get expert action
                action = expert_init_actions[t]
                task_completed = traj_data['plan']['low_actions'][t+1]['high_idx'] != traj_data['plan']['low_actions'][t]['high_idx']
                compressed_mask = action['args']['mask'] if 'mask' in action['args'] else None
                mask = env.decompress_mask(compressed_mask) if compressed_mask is not None else None

                # forward model
                if not args.skip_model_unroll_with_expert:
                    model.step(feat, prev_action=prev_action)
                    prev_action = action['action'] if not args.no_teacher_force_unroll_with_expert else None

                # execute expert action
                success, _, _, err, _ = env.va_interact(action['action'], interact_mask=mask, smooth_nav=args.smooth_nav, debug=args.debug)
                if not success:
                    print ("expert initialization failed")
                    break

                # update transition reward
                _, _ = env.get_transition_reward()

            # subgoal evaluation
            else:

                # forward model
                m_out = model.step(feat, prev_action=prev_action)
                m_pred = model.extract_preds(m_out, [traj_data], feat, clean_special_tokens=False)
                m_pred = list(m_pred.values())[0]

                # get action and mask
                action, mask = m_pred['action_low'], m_pred['action_low_mask'][0]
                if args.only_eval_mask:
                    try:
                        action = expert_init_actions_all[t]['discrete_action']['action']
                        if expert_init_actions_all[t]['high_idx'] > eval_idx:
                            break   # fail
                    except:
                        break
                mask = np.squeeze(mask, axis=0) if model.has_interaction(action) else None

                # debug
                if args.debug:
                    print("Pred: ", action)

                # update prev action
                prev_action = str(action)

                if action not in cls.TERMINAL_TOKENS:
                    # use predicted action and mask (if provided) to interact with the env
                    t_success, _, _, err, _ = env.va_interact(action, interact_mask=mask, smooth_nav=args.smooth_nav, debug=args.debug)
                    if not t_success:
                        fails += 1
                        if fails >= args.max_fails:
                            print("Interact API failed %d times" % (fails) + "; latest error '%s'" % err)
                            break

                # next time-step
                t_reward, t_done = env.get_transition_reward()
                reward += t_reward

                # update subgoals
                curr_subgoal_idx = env.get_subgoal_idx()
                if curr_subgoal_idx == eval_idx:
                    subgoal_success = True
                    break

                # terminal tokens predicted
                if action in cls.TERMINAL_TOKENS:
                    print("predicted %s" % action)
                    break

            # increment time index
            t += 1

        # metrics
        pl = float(t - len(expert_init_actions)) + 1 # +1 for last action
        expert_pl = len([ll for ll in traj_data['plan']['low_actions'] if ll['high_idx'] == eval_idx])

        s_spl = (1 if subgoal_success else 0) * min(1., expert_pl / (pl + sys.float_info.epsilon))
        plw_s_spl = s_spl * expert_pl

        # log success/fails
        lock.acquire()

        # results
        for sg in cls.ALL_SUBGOALS:
            results[sg] = {
                    'sr': 0.,
                    'successes': 0.,
                    'evals': 0.,
                    'sr_plw': 0.
            }

        log_entry = {'trial': traj_data['task_id'],
                     'type': traj_data['task_type'],
                     'repeat_idx': int(r_idx),
                     'subgoal_idx': int(eval_idx),
                     'subgoal_type': subgoal_action,
                     'subgoal_instr': subgoal_instr,
                     'subgoal_success_spl': float(s_spl),
                     'subgoal_path_len_weighted_success_spl': float(plw_s_spl),
                     'subgoal_path_len_weight': float(expert_pl),
                     'reward': float(reward)}
        if subgoal_success:
            sg_successes = successes[subgoal_action]
            sg_successes.append(log_entry)
            successes[subgoal_action] = sg_successes
        else:
            sg_failures = failures[subgoal_action]
            sg_failures.append(log_entry)
            failures[subgoal_action] = sg_failures

        # save results
        print("-------------")
        subgoals_to_evaluate = list(successes.keys())
        subgoals_to_evaluate.sort()
        for sg in subgoals_to_evaluate:
            num_successes, num_failures = len(successes[sg]), len(failures[sg])
            num_evals = len(successes[sg]) + len(failures[sg])
            if num_evals > 0:
                sr = float(num_successes) / num_evals
                total_path_len_weight = sum([entry['subgoal_path_len_weight'] for entry in successes[sg]]) + \
                                        sum([entry['subgoal_path_len_weight'] for entry in failures[sg]])
                sr_plw = float(sum([entry['subgoal_path_len_weighted_success_spl'] for entry in successes[sg]]) +
                                    sum([entry['subgoal_path_len_weighted_success_spl'] for entry in failures[sg]])) / total_path_len_weight

                results[sg] = {
                    'sr': sr,
                    'successes': num_successes,
                    'evals': num_evals,
                    'sr_plw': sr_plw
                }

                print("%s ==========" % sg)
                print("SR: %d/%d = %.3f" % (num_successes, num_evals, sr))
                print("PLW SR: %.3f" % (sr_plw))
        print("------------")

        lock.release()

    def create_stats(self):
        '''
        storage for success, failure, and results info
        '''
        self.successes, self.failures = self.manager.dict(), self.manager.dict()
        self.results = self.manager.dict()

    def save_results(self):
        results = {'successes': dict(self.successes),
                   'failures': dict(self.failures),
                   'results': dict(self.results)}

        save_path = os.path.dirname(self.args.model_path)
        save_path = os.path.join(save_path, 'subgoal_results_' + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + '.json')
        with open(save_path, 'w') as r:
            json.dump(results, r, indent=4, sort_keys=True)