#!/usr/bin/env python2.7

import argparse
import pickle
import lmdb
import caffe
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from data.utils import get_face_size
from data.utils import get_logger


plt.switch_backend('agg')
logger = get_logger()


def save_scores(scores, net):
  """save pos_scores and neg_scores to out
  """
  with open('tmp/%snet_scores'%net, 'w') as fout:
    pickle.dump(scores, fout)


def load_scores(net):
  """load pos_scores and neg_scores from out
  """
  with open('tmp/%snet_scores'%net, 'r') as fin:
    scores = pickle.load(fin)
  return scores


def plot_score(scores, net):
  """plot pos, neg scores
  """
  pos_train_scores, neg_train_scores, pos_val_scores, neg_val_scores = scores

  def plot(pos_scores, neg_scores, out, data_type='train', log=False):
    plt.figure()
    plt.title('%s distribution'%data_type)
    plt.xlabel('score')
    plt.hist(pos_scores, bins=100, histtype='step', log=log, normed=True, color='red')
    plt.hist(neg_scores, bins=100, histtype='step', log=log, normed=True, color='green')
    plt.savefig(out)

  plot(pos_train_scores, neg_train_scores, 'tmp/%snet_dis_train.png'%net, data_type='train', log=False)
  plot(pos_train_scores, neg_train_scores, 'tmp/%snet_dis_train_log.png'%net, data_type='train', log=True)
  plot(pos_val_scores, neg_val_scores, 'tmp/%snet_dis_val.png'%net, data_type='val', log=False)
  plot(pos_val_scores, neg_val_scores, 'tmp/%snet_dis_val_log.png'%net, data_type='val', log=True)

  # threshold for TP, TN
  for th in np.arange(0, 1, 0.02):
    logger.info('Threshold = %f'%th)
    logger.info("=== Train ===")
    tp = float(np.sum(pos_train_scores > th)) / len(pos_train_scores)
    tn = float(np.sum(neg_train_scores < th)) / len(neg_train_scores)
    logger.info('TP: %f'%tp)
    logger.info('TN: %f'%tn)
    logger.info("=== Train ===")
    logger.info("=== Val ===")
    tp = float(np.sum(pos_val_scores > th)) / len(pos_val_scores)
    tn = float(np.sum(neg_val_scores < th)) / len(neg_val_scores)
    logger.info('TP: %f'%tp)
    logger.info('TN: %f'%tn)
    logger.info("=== Val ===")


def main(args):
  # if preload
  if args.preload != 'none':
    net = args.preload
    scores = load_scores(net)
    plot_score(scores, net)
    return

  # use cnn to get scores
  net = args.net
  face_train = 'data/%snet_face_train'%net
  face_val = 'data/%snet_face_val'%net
  landmark_train = 'data/%snet_landmark_train'%net
  landmark_val = 'data/%snet_landmark_val'%net
  nonface_train = 'data/%snet_nonface_train'%net
  nonface_val = 'data/%snet_nonface_val'%net

  cnn = caffe.Net('proto/%s.prototxt'%net, caffe.TEST, weights='result/%s.caffemodel'%net)

  def eval_db(db_name):
    """eval cnn over db, return face prob score
    """
    logger.info(db_name)
    db = lmdb.open(db_name)
    face_size = get_face_size(net)
    face_shape = (3, face_size, face_size)
    batch_size = 256
    face_data = np.zeros((batch_size, 3, face_size, face_size), dtype=np.float32)
    cnn.blobs['data'].reshape(*face_data.shape)
    with db.begin() as txn:
      size = int(txn.get('size'))
      batches = size / batch_size
      scores = np.zeros(batch_size*batches, dtype=np.float32)
      for i in range(batches):
        if i%500 == 0:
          logger.info('process batch %d'%i)
        start = i*batch_size
        end = start + batch_size
        for j in range(start, end):
          key = '%08d_data'%j
          face_data[j-start] = np.fromstring(txn.get(key), dtype=np.uint8).reshape(face_shape).astype(np.float32)
        cnn.blobs['data'].data[...] = (face_data - 128) / 128
        cnn.forward()
        if net == 'p':
          scores[start:end] = cnn.blobs['face_prob'].data[:, 1, 0, 0]
        else:
          scores[start:end] = cnn.blobs['face_prob'].data[:, 1]
    db.close()
    return scores

  face_train_scores = eval_db(face_train)
  face_val_scores = eval_db(face_val)
  landmark_train_scores = eval_db(landmark_train)
  landmark_val_scores = eval_db(landmark_val)
  nonface_train_scores = eval_db(nonface_train)
  nonface_val_scores = eval_db(nonface_val)
  pos_train_scores = np.concatenate([face_train_scores, landmark_train_scores])
  pos_val_scores = np.concatenate([face_val_scores, landmark_val_scores])
  neg_train_scores = nonface_train_scores
  neg_val_scores= nonface_val_scores

  scores = (pos_train_scores, neg_train_scores, pos_val_scores, neg_val_scores)
  save_scores(scores, net)
  plot_score(scores, net)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--net', type=str, default='p', help='net type')
  parser.add_argument('--preload', type=str, default='none', help='already extracted net output')
  args = parser.parse_args()
  assert args.net in ['p', 'r', 'o']
  assert args.preload in ['none', 'p', 'r', 'o']
  main(args)
