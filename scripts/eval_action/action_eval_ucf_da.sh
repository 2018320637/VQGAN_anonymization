#!/bin/bash
cd ../..
CUDA_VISIBLE_DEVICES=5 python eval_action_da.py \
    --src 'ucf' \
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
    --action_weight 1.0 \
    --pseudo_weight 1.0 \
    --entropy_weight 0.0 \
    --domain_weight 1.0 \
    --self_pretrained_action '/home/zhiwei/source/SPAct/anonymization_training/action_da_logs_vqgan/train_action_da_ucf2hmdb_frames_16_bs_30_lr_0.005_optimizer_sgd_action_1.0_pseudo_1.0_entropy_0.0_domain_1.0_th_0.95/model_ucf2hmdb_90_acc_0.8472222222222222.pth' \
    --self_pretrained_pseudo '/home/zhiwei/source/SPAct/anonymization_training/action_da_logs_vqgan/train_action_da_ucf2hmdb_frames_16_bs_30_lr_0.005_optimizer_sgd_action_1.0_pseudo_1.0_entropy_0.0_domain_1.0_th_0.95/model_ucf2hmdb_90_acc_0.8472222222222222.pth' \
    --self_pretrained_fa '/home/zhiwei/source/SPAct/SPAct_baseline/logs/spact_minmax/SPAct_minmax_ucf2hmdb_frames_16_rate_2_bs_7_lr_ft_0.001_lr_fb_0.001_lr_fa_0.001_privacy_weight_0.5_action_weight_1.0/model_60_ucf_0.9476_hmdb_0.7556.pth'