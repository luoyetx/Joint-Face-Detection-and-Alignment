#!/usr/bin/env python2.7

import os
import argparse
import lmdb


def main(args):
  net_type = args.net
  if net_type == 'p':
    net = 'pnet'
  elif net_type == 'r':
    net = 'rnet'
  else:
    assert net_type == 'o'
    net = 'onet'

  def get_size(db_name):
    db = lmdb.open(db_name)
    txn = db.begin()
    size = int(txn.get('size'))
    txn.abort()
    db.close
    return size

  face_train_size = get_size('data/%s_face_train'%net)
  face_val_size = get_size('data/%s_face_val'%net)
  landmark_train_size = get_size('data/%s_landmark_train'%net)
  landmark_val_size = get_size('data/%s_landmark_val'%net)
  nonface_train_size = get_size('data/%s_nonface_train'%net)
  nonface_val_size = get_size('data/%s_nonface_val'%net)
  print 'face train size', face_train_size
  print 'face val size', face_val_size
  print 'landmark train size', landmark_train_size
  print 'landmark val size', landmark_val_size
  print 'nonface train size', nonface_train_size
  print 'nonface val size', nonface_val_size
  print ''

  base_batch_size = args.size
  face_batch_size = base_batch_size
  landmark_batch_size = base_batch_size
  nonface_batch_size = 4 * base_batch_size
  print 'face batch size', face_batch_size
  print 'face train epoch size', face_train_size / face_batch_size
  print 'face val epoch size', face_val_size / face_batch_size
  print 'landmark batch size', landmark_batch_size
  print 'landmark train epoch size', landmark_train_size / landmark_batch_size
  print 'landmark val epoch size', landmark_val_size / landmark_batch_size
  print 'nonface batch size', nonface_batch_size
  print 'nonface train epoch size', nonface_train_size / nonface_batch_size
  print 'nonface val epoch size', nonface_val_size / nonface_batch_size


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--net', type=str, default='p', help='net type')
  parser.add_argument('--size', type=int, default=256, help='base batch size')
  args = parser.parse_args()
  assert args.net in  ['p', 'r', 'o']
  main(args)
