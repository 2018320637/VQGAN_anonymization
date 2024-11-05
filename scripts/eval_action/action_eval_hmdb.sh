#!/bin/bash
cd ../..
CUDA_VISIBLE_DEVICES=4 python eval_action.py \
    --src 'hmdb' \
    --num_frames 16 \
    --sample_every_n_frames 2 \
    --num_workers 8 \
    --batch_size 64 \
    --learning_rate 0.001 \
    --num_epochs 100 \
    --val_freq 5 \
    --reso_h 128 \
    --reso_w 128 \
    --num_classes 12 \
    --self_pretrained_action '/home/zhiwei/source/SPAct/anonymization_training/action_da_logs_vqgan/train_action_da_hmdb2ucf_frames_16_bs_30_lr_0.005_optimizer_sgd_action_1.0_pseudo_1.0_entropy_1.0_domain_1.0/model_hmdb2ucf_48_acc_0.9667250437828371.pth' \
    --self_pretrained_fa '/home/zhiwei/source/SPAct/SPAct_baseline/logs/spact_minmax/SPAct_minmax_hmdb2ucf_frames_16_rate_2_bs_7_lr_ft_0.001_lr_fb_0.001_lr_fa_0.001_privacy_weight_0.5_action_weight_1.0/model_60_hmdb_0.8167_ucf_0.7552.pth'