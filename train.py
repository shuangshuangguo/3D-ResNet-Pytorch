import torch
from torch import nn
from torch.autograd import Variable
import time
import os
from validation import val_epoch
from model import generate_model
from opts import parse_opts
from utils import AverageMeter, accuracy, adjust_learning_rate, save_checkpoint
from dataset import Video
from spatial_transforms import (Compose, Normalize, Scale,
                                CenterCrop, CornerCrop, MultiScaleCornerCrop,
                                MultiScaleRandomCrop, RandomHorizontalFlip, ToTensor)
from temporal_transforms import TemporalRandomCrop, LoopPadding
import numpy as np
from torch.optim import lr_scheduler
from mean import get_mean
from collections import defaultdict
best_prec1 = 0


def train_epoch(epoch, data_loader, model, criterion, optimizer, opt):
    print('train at epoch {} with lr {}'.format(epoch, optimizer.param_groups[-1]['lr']))
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1  = AverageMeter()
    top5  = AverageMeter()
    end_time = time.time()
    for i, (inputs, targets, _) in enumerate(data_loader):
        data_time.update(time.time() - end_time)
        if not opt.no_cuda:
            targets = targets.cuda(async=True)
        inputs = Variable(inputs)
        targets_var = Variable(targets)
        outputs = model(inputs)
        loss = criterion(outputs, targets_var)
        prec1, prec5 = accuracy(outputs.data, targets, topk=(1, 5))
        losses.update(loss.data[0], inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        top5.update(prec5[0], inputs.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_time.update(time.time() - end_time)
        end_time = time.time()
        if i % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch, i + 1, len(data_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1, top5=top5))


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.eval()
    end = time.time()
    for i, (input, target, _) in enumerate(val_loader):
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)
        output = model(input_var)
        loss = criterion(output, target_var)
        prec1, prec5 = accuracy(output.data, target, topk=(1,5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        if i % opt.print_freq == 0:
            print(('Test: [{0}/{1}]\t'
                   'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                   'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                   'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5)))
    print(('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
       .format(top1=top1, top5=top5, loss=losses)))
    return losses.avg, top1.avg


if __name__ == '__main__':
    opt = parse_opts()
    opt.mean = get_mean(1)
    opt.arch = '{}-{}'.format(opt.model_name, opt.model_depth)
    opt.sample_duration = 16
    opt.scales = [opt.initial_scale]
    for i in range(1, opt.n_scales):
        opt.scales.append(opt.scales[-1] * opt.scale_step)
    print('#####', opt.scales)
    print(opt.mean)
    spatial_transform = Compose([MultiScaleCornerCrop(opt.scales, opt.sample_size),
                                 RandomHorizontalFlip(),
                                 ToTensor(1),
                                 Normalize(opt.mean, [1, 1, 1])])
    temporal_transform = TemporalRandomCrop(opt.sample_duration)
    train_data = Video(opt.train_list, spatial_transform=spatial_transform,
                 temporal_transform=temporal_transform,
                sample_duration=opt.sample_duration, n_samples_for_each_video=1)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=opt.batch_size,                                              shuffle=True, num_workers=opt.n_threads, pin_memory=True)

    val_spatial_transform = Compose([Scale(opt.sample_size),
                                 CenterCrop(opt.sample_size),
                                 ToTensor(1),
                                 Normalize(opt.mean, [1, 1, 1])])
    val_temporal_transform = LoopPadding(opt.sample_duration)
    val_data = Video(opt.val_list, spatial_transform=val_spatial_transform,
                 temporal_transform=val_temporal_transform,
                sample_duration=opt.sample_duration, n_samples_for_each_video=0)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=opt.batch_size,                                              shuffle=False, num_workers=opt.n_threads, pin_memory=True)
    model, policies = generate_model(opt)
    model = nn.DataParallel(model, device_ids=opt.gpus).cuda()

    if opt.finetune:
        if os.path.isfile(opt.finetune):
            print('finetuning from model {}'.format(opt.finetune))
            model_data = torch.load(opt.finetune)
            own_state = model.state_dict()
            for k, v in model_data['state_dict'].items():
                if 'fc' in k:
                    continue
                print(k)
                if isinstance(v, torch.nn.parameter.Parameter):
                    v = v.data
                assert v.dim() == own_state[k].dim(), '{} {} vs {}'.format(k, v.dim(), own_state[k].dim())
                own_state[k].copy_(v)
        else:
            assert False, ("=> no checkpoint found at '{}'".format(opt.finetune))
    if opt.resume:
        if os.path.isfile(opt.resume):
            print('loading model {}'.format(opt.resume))
            model_data = torch.load(opt.resume)
            opt.start_epoch = model_data['epoch']
            best_prec1 = model_data['best_prec1']
            model.load_state_dict(model_data['state_dict'])
        else:
            assert False,("=> no checkpoint found at '{}'".format(opt.resume))

    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(policies, opt.lr, momentum=opt.momentum, dampening=opt.dampening,
                                weight_decay=opt.weight_decay, nesterov=opt.nesterov)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=opt.lr_patience)
    for epoch in range(opt.start_epoch, opt.epochs):
        #adjust_learning_rate(optimizer, epoch, opt.lr_steps, opt)
        train_epoch(epoch, train_loader, model, criterion, optimizer, opt)
        if (epoch + 1) % opt.eval_freq == 0 or epoch == opt.epochs - 1:
            loss, prec1 = validate(val_loader, model, criterion)
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': opt.arch,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'best_prec1': best_prec1,
            }, is_best, opt.snapshot_pref)
            print('best_prec1: ', best_prec1)
        scheduler.step(loss)
