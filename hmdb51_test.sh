#!/bin/bash

python3 -u test.py --val_list splits/hmdb51_val_split_1.txt \
    --n_classes 51 --batch_size 128 \
    --model model/hmdb51_resnet18_16frame_finetune/hmdb51_resnet18_16frame_finetune_model_best_12_55.651074592985786.pth.tar \
    --result_path scores/hmdb51_resnet18_16frame_finetune_new.json
