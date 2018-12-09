#!/bin/bash

LOG="logs/ucf101_resnet18_16frame_finetune_112$RANK.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

python3 -u train.py --train_list splits/ucf101_train01.txt \
    --val_list splits/ucf101_test01.txt --n_classes 101 \
    --batch_size 128 --sample_size 112 \
    --lr 0.001  --epochs 300 \
    --snapshot_pref ucf101_resnet18_16frame_finetune_112 \
    --finetune pretrain/resnet-18-kinetics.pth


