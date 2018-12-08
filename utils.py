# -*- coding: utf-8 -*-
import numpy as np
import os
import torch
import shutil

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def adjust_learning_rate(optimizer, epoch, lr_steps, opt):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
    lr = opt.lr * decay
    decay = opt.weight_decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = decay * param_group['decay_mult']


def save_checkpoint(state, is_best, snapshot_pref, filename='checkpoint.pth.tar'):
    dirname = os.path.join('model', snapshot_pref)
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    filename = '_'.join((snapshot_pref, str(state['epoch']), filename))
    filename = os.path.join(dirname, filename)
    torch.save(state, filename)
    if is_best:
        best_name = '_'.join((snapshot_pref, 'model_best', str(state['epoch']),str(state['best_prec1']) + '.pth.tar'))
        best_name = os.path.join('model', snapshot_pref, best_name)
        shutil.copyfile(filename, best_name)

