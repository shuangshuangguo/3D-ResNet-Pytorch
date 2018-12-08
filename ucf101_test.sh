#!/bin/bash

python3 -u test.py --val_list splits/ucf101_val_split_1.txt \
    --n_classes 101 --batch_size 4 --sample_size 112 \
    --model pretrain/resnet-18-kinetics-ucf101_split1.pth \
    --result_path scores/ucf101_resnet18_16frame_given.json
