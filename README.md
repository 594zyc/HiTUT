# HiTUT

Official code for the ACL 2021 Findings paper "Yichi Zhang and Joyce Chai. Hierarchical Task Learning from Language Instructions with Unified Transformers and Self-Monitoring" [[arXiv](https://arxiv.org/abs/2106.03427)].


## Requirements

Install requirements:
```bash
$ pip install -r requirements.txt
```


Download the raw trajectory data, the preprocessed data, the pretrained Mask-RCNN models and our trained HiTUT model. 
```bash
$ sh download.sh
```

Setup the ALFRED root path:
```bash
$ export ALFRED_ROOT=$(pwd)
```


## Evaluation
Run the following command to evaluate our HiTUT model on the validation sets. "--max_high_fails 9" corresponds to a maximum allowed backtracking number of 8. 
```bash
python models/eval/eval_mmt.py --eval_path exp/Jan27-roberta-mix/noskip_lr_mix_all_E-xavier768d_L12_H768_det-sep_dp0.1_di0.1_step_lr5e-05_0.999_type_sd999 --ckpt model_best_seen.pth --gpu --max_high_fails 9 --max_fails 10 --eval_split valid_seen --eval_enable_feat_posture --num_threads 4 --name_temp eval_valid_seen
python models/eval/eval_mmt.py --eval_path exp/Jan27-roberta-mix/noskip_lr_mix_all_E-xavier768d_L12_H768_det-sep_dp0.1_di0.1_step_lr5e-05_0.999_type_sd999 --ckpt model_best_seen.pth --gpu --max_high_fails 9 --max_fails 10 --eval_split valid_unseen --eval_enable_feat_posture --num_threads 4 --name_temp eval_valid_unseen
```

Leaderboard evaluation: 
```bash
python models/eval/leaderboard.py --eval_path exp/Jan27-roberta-mix/noskip_lr_mix_all_E-xavier768d_L12_H768_det-sep_dp0.1_di0.1_step_lr5e-05_0.999_type_sd999 --ckpt model_best_seen.pth --gpu --max_high_fails 10 --eval_enable_feat_posture --num_threads 4 --name_temp eval_test
```


## Training
Run the following command to reproduce the multi-task training procedure for HiTUT. 
```bash
python models/train/train_mmt.py --gpu --use_bert --bert_model roberta --dropout 0.1 --drop_input 0.1 --enable_feat_posture --train_level mix --train_proportion 100 --valid_metric type --batch 84 --lr 5e-5 --focal_loss --emb_init xavier --emb_dim 768  --bert_lr_schedule --early_stop 2 --seed 999 --low_data all --exp_temp YOUR_EXP_PATH_NAME  --name_temp YOUR_EXP_NAME
```

## Contact
Feel free to create an issue or send email to zhangyic@umich.edu