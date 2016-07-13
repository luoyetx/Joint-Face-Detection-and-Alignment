#!/usr/bin/env python2.7

import os
import argparse
import lmdb
import numpy as np
from data.utils import get_logger


def main(args):
  net_type = args.net
  if net_type == 'p':
    net = 'pnet'
  elif net_type == 'r':
    net = 'rnet'
  else:
    assert net_type == 'o'
    net = 'onet'
  logger = get_logger()
  db = lmdb.open('data/%s_nonface_train'%net)
  with db.begin() as txn:
    size = int(txn.get('size'))
    logger.info('random read')
    for i in np.random.permutation(size):
      face_key = '%08d_data'%i
      offset_key = '%08d_offset'%i
      txn.get(face_key)
      txn.get(offset_key)
    logger.info('done')
    logger.info('sequential read')
    for i in range(size):
      face_key = '%08d_data'%i
      offset_key = '%08d_offset'%i
      txn.get(face_key)
      txn.get(offset_key)
    logger.info('done')
  db.close()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--net', type=str, default='p', help='net type')
  args = parser.parse_args()
  main(args)
