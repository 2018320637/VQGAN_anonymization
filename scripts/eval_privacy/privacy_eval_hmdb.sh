#!/bin/bash
cd ../..
CUDA_VISIBLE_DEVICES=4 python privacy_train.py \
    --src 'hmdb' \
    --tar 'ucf' \
    --architecture 'resnet50' \
    --train_backbone \
    --num_frames 16 \
    --sample_every_n_frames 2 \
    --num_workers 8 \
    --batch_size 64 \
    --learning_rate 0.005 \
    --num_epochs 100 \
    --val_freq 5 \
    --reso_h 128 \
    --reso_w 128 \
    --num_classes 5 \
    --self_pretrained_privacy '/home/zhiwei/source/SPAct/anonymization_training/experiments/privacy_logs/train_privacy_resnet50_hmdb_frames_16_rate_2_bs_64_lr_0.005/model_hmdb_55_cMAP_0.6512181795908678.pth' \
    --self_pretrained_fa '/home/zhiwei/source/SPAct/SPAct_baseline/logs/spact_minmax/SPAct_minmax_hmdb2ucf_frames_16_rate_2_bs_7_lr_ft_0.001_lr_fb_0.001_lr_fa_0.001_privacy_weight_0.5_action_weight_1.0/model_60_hmdb_0.8167_ucf_0.7552.pth'