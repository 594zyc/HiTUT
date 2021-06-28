import logging, os, time


def Config(parser):
        # general settings
    parser.add_argument('--seed', default=123, help='random seed', type=int)
    parser.add_argument('--raw_data', default='data/full_2.1.0', help='dataset folder')
    parser.add_argument('--pp_data', default='data/full_2.1.0_pp', help='folder name for preprocessed data')
    parser.add_argument('--img_data', default='data/full_2.1.0_imgs.hdf5', help='path of the hdf5 data file storing the images of full_2.1.0')
    parser.add_argument('--splits', default='data/splits/oct21.json', help='json file containing train/dev/test splits')
    parser.add_argument('--all_detector_path', default='models/detector/maskrcnn_all.pth',
        help='path of loading pretrained mask rcnn model for all 105 objects')
    parser.add_argument('--obj_detector_path', default='models/detector/mrcnn_object.pth',
        help='path of loading pretrained mask rcnn model for all 73 movable objects')
    parser.add_argument('--rec_detector_path', default='models/detector/mrcnn_receptacle.pth',
        help='path of loading pretrained mask rcnn model for all 32 static receptacles')

    parser.add_argument('--preprocess', action='store_true', help='store preprocessed data to json files')

    parser.add_argument('--exp_temp', default='exp', help='temp experimental sub directory for saving models and logs')
    parser.add_argument('--name_temp', default='exp', help='temp experimental name for saving models and logs')
    parser.add_argument('--use_templated_goals', help='use templated goals instead of human-annotated goal descriptions', action='store_true')
    parser.add_argument('--image_size', default=300, type=int, help='image pixel size (assuming square shape eg: 300x300)')
    parser.add_argument('--vocab_size', default=1500, type=int, help='vocabulary size')


    # model settings
    parser.add_argument('--max_enc_length', default=160, type=int, help='maximum length of encoder input')
    parser.add_argument('--use_bert', action='store_true', help='use a pretrained bert model as the encoder')
    parser.add_argument('--bert_model', default='bert', choices=['bert', 'albert', 'mobilebert', 'roberta'], help='which pretrained bert to use')
    parser.add_argument('--bert_lr_schedule', action='store_true', help='use a warmup-linear-decay lr scheduler for bert')
    parser.add_argument('--enc_layer_num', default=4, type=int)
    parser.add_argument('--head_num', default=4, type=int)
    parser.add_argument('--hidden_dim', default=512, type=int)
    parser.add_argument('--inner_dim', default=2048, type=int)
    parser.add_argument('--dropout', default=0.0, type=float, help='dropout rate for self-attention block hidden states')
    parser.add_argument('--drop_input', default=0.0, type=float, help='dropout rate for input feats')
    parser.add_argument('--topk_objs', default=30, type=int, help='use top k object proposals detected by the object detector module')
    parser.add_argument('--confidence', default=0.4, help='object proposals lower than the confidence score wil be pruned', type=float)
    parser.add_argument('--detector_type', default='sep', type=str, choices=['all', 'sep'],
        help='use a single mrcnn model to detect all the objects or two mrcnn models to detect movable objects and static receptacles separately')
    parser.add_argument('--lang_max_length', default=64, type=int, help='cutoff to language input to lang_max_length')
    parser.add_argument('--history_max_length', default=32, type=int, help='only keep the history_max_length most recent actions')
    parser.add_argument('--emb_dim', default=300, type=int, help='word embedding size')
    parser.add_argument('--emb_init', default='xavier', help='word embedding initialization weights')
    parser.add_argument('--emb_freeze',  action='store_true', help='freeze word embedding')
    parser.add_argument('--pred_head_pos', default='cls', type=str, choices=['cls', 'sep'],
        help='To use the first [CLS] output to make all predictions [cls] or use the final positions to of actions [sep]')

    # training settings
    parser.add_argument('--train_level', default='mix', type=str, choices=['mix', 'low', 'high'],
        help='train the model on low-level data only, high-level data only or mixed or both')
    parser.add_argument('--train_proportion', default=100, type=int, help='percentage of training data to use')
    parser.add_argument('--train_one_shot', action='store_true', help='use one-shot seed data to train')
    parser.add_argument('--valid_metric', default='type', type=str, choices=['type','arg', 'mask'],
        help='validation metric to select the best model')
    parser.add_argument('--low_data', default='all', type=str, choices=['all', 'mani', 'navi'],
        help='train the model on low-level data only, high-level data only or mixed or both')
    parser.add_argument('--resume', help='load a checkpoint')
    parser.add_argument('--batch', help='batch size', default=512, type=int)
    parser.add_argument('--epoch', help='number of epochs', default=50, type=int)
    parser.add_argument('--early_stop', help='validation check fail time before early stop training', default=5, type=int)
    parser.add_argument('--optimizer', default='adam', type=str, choices=['adam'], help='optimizer type')
    parser.add_argument('--weigh_loss', action='store_true',
        help='weigh each loss term based on its uncertainty. Credit to Kendall et al CVPR18 paper')
    parser.add_argument('--focal_loss', help='use focal loss', action='store_true')
    parser.add_argument('--focal_gamma', default=2, type=float, help='gamma in focal loss')
    parser.add_argument('--beta2', default=0.999, type=float, help='beta2 for adam optimizer')
    parser.add_argument('--lr', default=1e-4, type=float, help='optimizer learning rate')
    parser.add_argument('--lr_scheduler', default='step', type=str, choices=['step', 'noam'], help='lr_scheduler type')
    parser.add_argument('--step_decay_epoch', default=2, type=int, help='num epoch to step decay learning rate')
    parser.add_argument('--step_decay_factor', default=0.5, type=float, help='decay factor of learning rate each step')
    parser.add_argument('--noam_lr_factor', default=0.1, type=float,
        help='optimizer learning rate factor for noam')
    parser.add_argument('--noam_warmup_iter', default=2000, type=int,
        help='warmup iteration number for step/noam')

    parser.add_argument('--auxiliary_loss_navi', help='additional navigation loss', action='store_true')
    parser.add_argument('--random_skip', help='random skip some data of each epoch', action='store_true')
    parser.add_argument('--disable_feat_lang', help='do not use language features as input', action='store_true')
    parser.add_argument('--disable_feat_vis', help='do not use visual features as input', action='store_true')
    parser.add_argument('--disable_feat_action_his', help='do not use action history features as input', action='store_true')
    parser.add_argument('--enable_feat_vis_his', help='use additional history visual features as input', action='store_true')
    parser.add_argument('--enable_feat_posture', help='use additional agent posture features as input', action='store_true')

    parser.add_argument('--num_threads',  default=0, type=int, help='enable multi-threading parallelism for data preprocessing if num_thread >0')
    parser.add_argument('--gpu', help='use gpu', action='store_true')
    parser.add_argument('--fast_epoch', action='store_true', help='fast epoch during debugging')
    parser.add_argument('--debug', dest='debug', action='store_true')

    # model_structure_parameters = [
    #     'enc_layer_num', 'head_num', 'hidden_dim', 'emb_dim', 'train_level', 'low_data', 'inner_dim',
    # ]

    # return model_structure_parameters
