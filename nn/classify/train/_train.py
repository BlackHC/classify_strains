#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 18:50:59 2017

@author: ajaver
"""
import tensorflow as tf
from sklearn.metrics import f1_score

import atexit
import torch
import os
import shutil
import tqdm
import warnings
import numpy as np
from collections import OrderedDict
from ..data import get_datset_file, get_valid_strains, \
    SkeletonsFlowFull, SkeletonsFlowShuffled, IS_CUDA


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))

    # calculate the global f1 score
    # prefer to use scikit instead of having to program it again in torch
    ytrue = target.data.cpu().numpy()
    ypred = pred.data[0].cpu().numpy()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f1 = f1_score(ytrue, ypred, average='macro')
    return res, f1


def save_checkpoint(state, is_best, save_dir, filename='checkpoint.pth.tar'):
    checkpoint_path = os.path.join(save_dir, filename)
    torch.save(state, checkpoint_path)
    if is_best:
        best_path = os.path.join(save_dir, 'model_best.pth.tar')
        shutil.copyfile(checkpoint_path, best_path)


class TBLogger(object):
    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer = tf.summary.FileWriter(log_dir)
        atexit.register(self.writer.close)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        summary = tf.Summary(
            value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)


class Trainer(object):
    def __init__(self,
                 model,
                 optimizer,
                 criterion,
                 train_generator,
                 val_generator,
                 n_epochs,
                 log_dir
                 ):

        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_generator = train_generator
        self.val_generator = val_generator
        self.n_epochs = n_epochs
        self.epoch = None

        # Set the logger
        self.log_dir = log_dir
        self.logger = TBLogger(log_dir)

    def fit(self):
        best_prec1 = 0
        for self.epoch in range(1, self.n_epochs + 1):
            train_metrics = self._train_epoch()
            # save train metrics
            for tag, value in train_metrics.items():
                self.logger.scalar_summary(tag, value, self.epoch)

            # save validation metrics
            val_metrics = self._val_epoch()

            val_pred1 = val_metrics['val_pred1']
            is_best = val_pred1 > best_prec1
            best_prec1 = max(val_pred1, best_prec1)

            state = {
                'epoch': self.epoch,
                'state_dict': self.model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': self.optimizer.state_dict(),
            }
            save_checkpoint(state, is_best, save_dir=self.log_dir)
            for tag, value in val_metrics.items():
                self.logger.scalar_summary(tag, value, self.epoch)

    def _train_epoch(self):
        self.model.train()
        pbar = tqdm.tqdm(self.train_generator)

        generator = self.train_generator

        all_metrics = []
        for input_var, target_var in pbar:
            snps = torch.autograd.Variable(torch.from_numpy(
                generator.snps_data[generator.strain_codes.loc[
                    target_var.data.numpy()].strain].values.T.astype(
                    np.float32)))
            output = self.model(input_var, snps)
            loss = self.criterion(output, target_var)

            # Before the backward pass, use the optimizer object to zero all
            # of the
            # gradients for the variables it will update (which are the
            # learnable weights
            # of the model)
            self.optimizer.zero_grad()
            # Use autograd to compute the backward pass. This call will
            # compute the
            # gradient of loss with respect to all Variables with
            # requires_grad=True.
            # After this call w1.grad and w2.grad will be Variables holding
            # the gradient
            # of the loss with respect to w1 and w2 respectively.
            loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem
            # in RNNs / LSTMs.
            torch.nn.utils.clip_grad_norm(self.model.parameters(), 0.25)
            self.optimizer.step()

            m = self._metrics(output[0], target_var, loss, is_train=True)
            all_metrics.append(m)
            pbar.set_description(self._pbar_description(m, is_train=True))

        return self._metrics_avg(all_metrics)

    def _val_epoch(self):
        self.model.eval()
        pbar = tqdm.tqdm(self.val_generator)
        generator = self.val_generator

        all_metrics = []
        for input_var, target_var in pbar:
            snps = torch.autograd.Variable(torch.from_numpy(
                generator.snps_data[generator.strain_codes.loc[
                    target_var.data.numpy()].strain].values.T.astype(
                    np.float32)))
            output = self.model(input_var, snps)
            loss = self.criterion(output, target_var)

            m = self._metrics(output[0], target_var, loss, is_train=False)
            all_metrics.append(m)
            pbar.set_description(self._pbar_description(m, is_train=False))
        return self._metrics_avg(all_metrics)

    def _pbar_description(self, m, is_train):
        m_str = ', '.join(['{}: {:.3f}'.format(*x) for x in m])
        d_str = 'Epoch : {} | {}'.format(self.epoch, m_str)
        if not is_train:
            return 'Val ' + d_str

        return d_str

    def _metrics(self, output, target_var, loss, is_train):
        (prec1, prec5), f1 = accuracy(output, target_var, topk=(1, 5))
        if is_train:
            prefix = 'train_'
        else:
            prefix = 'val_'

        tb = [('loss', loss.data[0]),
              ('pred1', prec1.data[0]),
              ('pred5', prec5.data[0]),
              ('f1', f1)
              ]
        tb = [(prefix + x, y) for x, y in tb]

        return tb

    def _metrics_avg(self, m):
        dd = [list(zip(*x)) for x in zip(*m)]
        return OrderedDict([(x[0][0], np.mean(x[1])) for x in dd])


def get_params_str(dataset, is_reduced, params):
    dd = ('R_' if is_reduced else '',
          dataset,
          params['transform_type'],
          '_N' if params['is_normalized'] else '',
          params['sample_size_seconds'],
          params['sample_frequency_s']
          )
    postfix = '_{}{}_{}{}_S{}_F{:.2}'.format(*dd)
    return postfix


def init_generator(dataset='',
                   data_file=None,
                   is_reduced=False,
                   sample_size_seconds=10,
                   sample_frequency_s=0.04,
                   n_batch=32,
                   transform_type='angles',
                   is_normalized=False,
                   _valid_strains=None  # used for testing
                   ):
    assert dataset in ['SWDB', 'CeNDR']

    if data_file is None:
        data_file = get_datset_file(dataset)

    if _valid_strains is None:
        valid_strains = get_valid_strains(dataset, is_reduced=True)
    else:
        valid_strains = _valid_strains

    gen_params = dict(
        data_file=data_file,
        n_batch=n_batch,
        valid_strains=valid_strains,
        sample_size_seconds=sample_size_seconds,
        sample_frequency_s=sample_frequency_s,
        transform_type=transform_type,
        is_normalized=is_normalized,
        is_torch=True
    )

    gen_details = get_params_str(dataset, is_reduced, gen_params)
    train_generator = SkeletonsFlowShuffled(set_type='train', **gen_params)
    val_generator = SkeletonsFlowFull(set_type='test', **gen_params)

    return gen_details, train_generator, val_generator
