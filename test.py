import torch
from torch import nn
from torch.autograd import Variable
from dataset import Video
from spatial_transforms import (Compose, Normalize, Scale, CenterCrop, CornerCrop, MultiScaleCornerCrop,
                                MultiScaleRandomCrop, RandomHorizontalFlip, ToTensor)
from temporal_transforms import LoopPadding, TemporalRandomCrop
import numpy as np
import time
import os
import json
from model import generate_model
from opts import parse_opts
from mean import get_mean
from collections import defaultdict
from utils import AverageMeter
from sklearn.metrics import confusion_matrix

def calculate_video_results(output_buffer, video_id, test_results):
    video_outputs = torch.stack(output_buffer)
    average_scores = torch.mean(video_outputs, dim=0)
    sorted_scores, locs = torch.topk(average_scores, k=10)
    video_results = []
    for i in range(sorted_scores.size(0)):
        video_results.append({'label': locs[i], 'score': sorted_scores[i]})
    test_results['results'][video_id] = video_results


def test(data_loader, model, opt):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end_time = time.time()
    test_results = {'results': {}}
    total_out = defaultdict(list)
    label_dict = {}
    for i, (inputs, targets, video_path) in enumerate(data_loader):
        data_time.update(time.time() - end_time)
        inputs = Variable(inputs, volatile=True)
        outputs = model(inputs)
        for j in range(outputs.size(0)):
            total_out[video_path[j]].append(outputs[j].data.cpu())
            label_dict[video_path[j]] = targets[j]

        batch_time.update(time.time() - end_time)
        end_time = time.time()
        print('[{}/{}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'.format(
                  i + 1, len(data_loader), batch_time=batch_time, data_time=data_time))
    for video_path, out in total_out.items():
        #print('##', out)
        calculate_video_results(out, video_path, test_results)

    pred, labels = [], []
    for video_path, video_result in test_results['results'].items():
        pred.append(video_result[0]['label'])
        labels.append(label_dict[video_path])
    cf = confusion_matrix(labels, pred).astype(float)
    cls_cnt = cf.sum(axis=1)
    cls_hit = np.diag(cf)
    cls_acc = cls_hit / cls_cnt
    print(cls_acc)
    print('Accuracy {:.02f}%'.format(np.mean(cls_acc) * 100))
    with open(opt.result_path, 'w') as f:
        json.dump(test_results, f)

if __name__ == '__main__':
    opt = parse_opts()
    opt.mean = get_mean()
    opt.arch = '{}-{}'.format(opt.model_name, opt.model_depth)
    opt.sample_duration = 16
    spatial_transform = Compose([Scale(opt.sample_size),
                                 CenterrCrop(opt.sample_size),
                                 ToTensor(1),
                                 Normalize(opt.mean, [1, 1, 1])])
    temporal_transform = LoopPadding(opt.sample_duration)
    data = Video(opt.val_list, spatial_transform=spatial_transform,
                 temporal_transform=temporal_transform,
                 sample_duration=opt.sample_duration, n_samples_for_each_video=0)
    data_loader = torch.utils.data.DataLoader(data, batch_size=opt.batch_size,
                                              shuffle=False, num_workers=opt.n_threads, pin_memory=True)
    model, _ = generate_model(opt)
    model = nn.DataParallel(model, device_ids=opt.gpus).cuda()
    print('loading model {}'.format(opt.model))
    model_data = torch.load(opt.model)
    assert opt.arch == model_data['arch']
    model.load_state_dict(model_data['state_dict'])
    model.eval()
    test(data_loader, model, opt)

