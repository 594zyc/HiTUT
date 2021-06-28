import os, sys, argparse, json, copy, logging

import torch
import torch.multiprocessing as mp

sys.path.append(os.path.join(os.environ['ALFRED_ROOT']))

from data.dataset import AlfredDataset
from models.config.configs import Config
from models.model.mmt import MultiModalTransformer
from models.nn.mrcnn import MaskRCNNDetector
from models.eval.eval_task import EvalTaskMMT
from models.eval.eval_subgoals import EvalSubgoalsMMT



def check_input(model, arg):
    for feat_arg in ['disable_feat_lang_high','disable_feat_lang_navi','disable_feat_lang_mani',
                             'disable_feat_vis', 'disable_feat_action_his',
                            'enable_feat_vis_his', 'enable_feat_posture']:

        feat_arg_eval = 'eval_'+ feat_arg
        if 'lang' in feat_arg:
            feat_arg = feat_arg[:-5]
        if not hasattr(model.args, feat_arg) or getattr(model.args, feat_arg) != getattr(arg, feat_arg_eval):
            logging.warning('WARNING: dismatch input option: %s'%feat_arg_eval)

if __name__ == '__main__':
    core_mask_op = "taskset -pc %s %d" %('0-40', os.getpid())
    os.system(core_mask_op)

    # multiprocessing settings
    mp.set_start_method('spawn')
    manager = mp.Manager()

    # parser
    parser = argparse.ArgumentParser()
    Config(parser)

    # eval settings
    parser.add_argument('--reward_config', default='models/config/rewards.json')
    parser.add_argument('--eval_split', type=str, default='valid_seen', choices=['valid_seen', 'valid_unseen'])
    parser.add_argument('--eval_path', type=str, default="exp/something")
    parser.add_argument('--ckpt_name', type=str, default="model_best_seen.pth")
    parser.add_argument('--num_core_per_proc', type=int, default=5, help='cpu cores used per process')
    # parser.add_argument('--model', type=str, default='models.model.seq2seq_im_mask')
    parser.add_argument('--subgoals', type=str, help="subgoals to evaluate independently, eg:all or GotoLocation,PickupObject...", default="")
    parser.add_argument('--smooth_nav', dest='smooth_nav', action='store_true', help='smooth nav actions (might be required based on training data)')
    parser.add_argument('--max_high_steps', type=int, default=20, help='max steps before a high-level episode termination')
    parser.add_argument('--max_high_fails', type=int, default=5, help='max failing times to try high-level proposals')
    parser.add_argument('--max_fails', type=int, default=999, help='max failing times in ALFRED benchmark')
    parser.add_argument('--max_low_steps', type=int, default=50, help='max steps before a low-level episode termination')
    parser.add_argument('--only_eval_mask', dest='only_eval_mask', action='store_true')
    parser.add_argument('--use_gt_navigation', dest='use_gt_navigation', action='store_true')
    parser.add_argument('--use_gt_high_action', dest='use_gt_high_action', action='store_true')
    parser.add_argument('--use_gt_mask', dest='use_gt_mask', action='store_true')
    parser.add_argument('--save_video', action='store_true')

    parser.add_argument('--eval_disable_feat_lang_high', help='do not use language features as high input', action='store_true')
    parser.add_argument('--eval_disable_feat_lang_navi', help='do not use language features as low-navi input', action='store_true')
    parser.add_argument('--eval_disable_feat_lang_mani', help='do not use language features as low-mani input', action='store_true')
    parser.add_argument('--eval_disable_feat_vis', help='do not use visual features as input', action='store_true')
    parser.add_argument('--eval_disable_feat_action_his', help='do not use action history features as input', action='store_true')
    parser.add_argument('--eval_enable_feat_vis_his', help='use additional history visual features as input', action='store_true')
    parser.add_argument('--eval_enable_feat_posture', help='use additional agent posture features as input', action='store_true')


    # parse arguments
    args = parser.parse_args()
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

    # setup model2
    # args2 = copy.deepcopy(args)
    # # args2.eval_path = 'exp/Jan23-low-navi/ori+pos_low_navi_E-xavier256d_L4_H512_det-sep_dp0.2_di0.2_step_lr1e-04_0.999_type_sd123'
    # # args2.eval_path = 'exp/Jan23-low-navi-sep/cls+aux+pos+vishis_low_navi_E-xavier256d_L4_H512_det-sep_dp0.2_di0.2_step_lr1e-04_0.999_type_sd123'
    # args2.eval_path = 'exp/Jan23-low-navi/ori+pos+vishis_low_navi_E-xavier256d_L4_H512_det-sep_dp0.2_di0.2_step_lr1e-04_0.999_type_sd123'
    # args2.ckpt_name = 'model_best_valid.pth'
    # args2_model = argparse.Namespace(**json.load(open(os.path.join(args2.eval_path, 'config.json'), 'r')))
    # ckpt_path2 = os.path.join(args2.eval_path, args2.ckpt_name)
    # ckpt2 = torch.load(ckpt_path2, map_location=device)
    # model2 = MultiModalTransformer(args2_model, alfred_data)
    # model2.load_state_dict(ckpt2)   #
    # model2.to(model2.device)

    # models = {'navi': model2, 'mani': model}


    # log dir
    eval_type = 'task' if not args.subgoals else 'subgoal'
    gt_navi = '' if not args.use_gt_navigation else '_gtnavi'
    gt_sg = '' if not args.use_gt_high_action else '_gtsg'
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
    log_name = '%s_%s_%s_maxfail%d_%s%s%s.log'%(args.name_temp, eval_type, args.eval_split, args.max_high_fails,
        input_str, gt_navi, gt_sg)
    if args.debug:
        log_name = log_name.replace('.log', '_debug.log')
    if isinstance(models, dict):
        log_name = log_name.replace('.log', '_sep.log')
    args.log_dir = os.path.join(args.eval_path, log_name)

    log_level = logging.DEBUG if args.debug else logging.INFO
    log_handlers = [logging.StreamHandler(), logging.FileHandler(args.log_dir)]
    logging.basicConfig(handlers=log_handlers, level=log_level,
        format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    if isinstance(models, dict):
        logging.info('model 1: %s'%ckpt_path)
        check_input(model, args)
        logging.info('model 2: %s'%ckpt_path2)
        check_input(model2, args2)
    else:
        logging.info('model: %s'%ckpt_path)
        check_input(model, args)

    # setup object detector
    detector = MaskRCNNDetector(args, detectors=[args.detector_type])

    # eval mode
    if args.subgoals:
        eval = EvalSubgoalsMMT(args, alfred_data, models, detector, manager)
    else:
        eval = EvalTaskMMT(args, alfred_data, models, detector, manager)


    # start threads
    # eval.run(model, detector, vocabs, task_queue, args, lock, stats, results)
    eval.start()