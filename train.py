#!/usr/bin/env python2.7
# coding = utf-8

import logging
import argparse
import numpy as np
import mxnet as mx
from mxnet.metric import check_label_shapes
from model import get_model, add_loss
from dataiter import MTDataIter


logging.basicConfig(level=logging.INFO)


class MTMetric(mx.metric.EvalMetric):
  """eval metric for network
  """

  def __init__(self, ws=[1.0, 1.0, 1.0]):
    self.ws = ws
    self.face_metric = mx.metric.CrossEntropy()
    self.bbox_metric = mx.metric.MSE()
    self.landmark_metric = mx.metric.MSE()
    super(MTMetric, self).__init__('fda', 3)

  def update(self, labels, preds):
    self.face_metric.update([labels[0]], [preds[0]])
    self.bbox_metric.update([labels[1]], [preds[1]])
    self.landmark_metric.update([labels[2]], [preds[2]])

  def reset(self):
    super(MTMetric, self).reset()
    self.face_metric.reset()
    self.bbox_metric.reset()
    self.landmark_metric.reset()

  def get(self):
    fm_name, fm_value = self.face_metric.get()
    bm_name, bm_value = self.bbox_metric.get()
    lm_name, lm_value = self.landmark_metric.get()
    tm_name = 'all loss'
    tm_value = self.ws[0]*fm_value + \
               self.ws[1]*bm_value + \
               self.ws[2]*lm_value
    return ([tm_name, fm_name, bm_name, lm_name],
            [tm_value, fm_value, bm_value, lm_value])


def get_data(args):
  """get dataiter of train data and val data
  """
  # net type
  net_type = args.net_type
  # epoch size of train data and val data
  # need to specificly detected by calculating data size of each data type
  train_epoch_size = args.train_epoch_size
  val_epoch_size = args.val_epoch_size
  # batch size of different data type
  face_batch_size = args.face_batch_size
  landmark_batch_size = args.landmark_batch_size
  nonface_batch_size = args.nonface_batch_size
  batch_sizes = [face_batch_size, landmark_batch_size, nonface_batch_size]

  train_data = MTDataIter(net_type=net_type,
                          is_train=True,
                          shuffle=True,
                          epoch_size=train_epoch_size,
                          batch_sizes=batch_sizes)
  val_data = MTDataIter(net_type=net_type,
                        is_train=False,
                        shuffle=True,
                        epoch_size=val_epoch_size,
                        batch_sizes=batch_sizes)
  return train_data, val_data


def get_ws(args):
  """get weights
  """
  net_type = args.net_type
  if net_type == 'p':
    ws = [1.0, 0.5, 0.5]
  elif net_type == 'r':
    ws = [1.0, 0.5, 0.5]
  else:
    assert net_type == 'o'
    ws = [1.0, 0.5, 1.0]
  return ws


def get_symbol(args):
  """get symbol of net_type
  """
  net_type = args.net_type
  net = get_model(net_type, is_train=True)
  ws = get_ws(args)
  net_with_loss = add_loss(net, ws=ws)
  return net_with_loss


def main(args):
  """main
  """
  train_data, val_data = get_data(args)
  symbol = get_symbol(args)

  devs = mx.cpu() if args.gpus is None else [mx.gpu(int(i)) for i in args.gpus.split(',')]
  lr_scheduler = mx.lr_scheduler.FactorScheduler(
    step=args.lr_reduce_step,
    factor=args.lr_reduce_factor,
    stop_factor_lr=args.lr_minimum)
  batch_size = args.face_batch_size + \
               args.landmark_batch_size + \
               args.nonface_batch_size
  checkpoint = mx.callback.do_checkpoint(args.model_save_prefix)
  kv = 'local'
  ws = get_ws(args)

  model = mx.model.FeedForward(
    ctx                 = devs,
    symbol              = symbol,
    num_epoch           = args.max_epoch,
    learning_rate       = args.lr,
    momentum            = args.momentum,
    wd                  = args.weight_decay,
    lr_scheduler         = lr_scheduler,
    initializer         = mx.init.Xavier(factor_type="in", magnitude=2.34))
  model.fit(
    X                   = train_data,
    eval_data           = val_data,
    eval_metric         = MTMetric(ws),
    kvstore             = kv,
    batch_end_callback  = mx.callback.Speedometer(batch_size, 200),
    epoch_end_callback  = checkpoint)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--net_type', type=str, default='p', help='network type, must be p or r or o')
  parser.add_argument('--face_batch_size', type=int, default=256, help='face data size in a batch')
  parser.add_argument('--landmark_batch_size', type=int, default=256, help='landmark data size in a batch')
  parser.add_argument('--nonface_batch_size', type=int, default=512, help='nonface data size in a batch')
  parser.add_argument('--train_epoch_size', type=int, default=625, help='how many batches in one train epoch')
  parser.add_argument('--val_epoch_size', type=int, default=156, help='how many batches in one val epoch')
  parser.add_argument('--max_epoch', type=int, default=200, help='number of epoches to train')
  parser.add_argument('--lr', type=float, default=0.01, help='base learning rate')
  parser.add_argument('--lr_reduce_step', type=int, default=625, help='number of batches when a reduction happens')
  parser.add_argument('--lr_reduce_factor', type=float, default=0.5, help='lr reduce factor')
  parser.add_argument('--lr_minimum', type=float, default=1e-5, help='minimun learning rate')
  parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
  parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay')
  parser.add_argument('--model_save_prefix', type=str, default='result/pnet', help='model save prefix')
  parser.add_argument('--gpus', type=str, help='gpus to use, e.g. "0,1,2"')
  args = parser.parse_args()
  main(args)
