#!/usr/bin/env python2.7
# pylint: disable=bad-indentation, no-member, invalid-name, line-too-long

import os
import shutil
import random
import argparse
import multiprocessing as mp
import cv2
import h5py
import caffe
import numpy as np
from caffe.proto import caffe_pb2
from google.protobuf import text_format
from jfda.config import cfg
from jfda.utils import load_celeba, get_logger, crop_face


logger = get_logger()


def fill_queues(data, qs):
  data_n = len(data)
  queue_n = len(qs)
  for i in range(len(data)):
    qs[i%queue_n].put(data[i])

def remove(f):
  if os.path.exists(f):
    os.remove(f)


# =========== prepare data ================

def lnet_reader_func(q_in, q_out):
  counter = 0
  while not q_in.empty():
    item = q_in.get()
    counter += 1
    if counter%10000 == 0:
      logger.info('%s reads %d', mp.current_process().name, counter)
    img_path, bbox, landmark = item
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
      logger.warn('read %s failed', img_path)
      continue
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    # assert w == h, 'bbox is not a square'
    landmark = landmark.reshape((5, 2))
    for _ in range(cfg.LNET_SAMPLE_PER_FACE):
      offset = np.random.rand(5, 2).astype(np.float32)
      offset = (2*offset - 1) * cfg.SAMPLE_RADIUS
      for scale in cfg.LNET_FACE_SCALES:
        l = w * scale
        target = offset.copy()
        # target = target * w / l
        target /= scale
        target = target.reshape(10)
        data = np.zeros((24, 24, 15), dtype=np.uint8)
        for i in range(5):
          x, y = landmark[i]
          x_offset, y_offset = offset[i] * w
          x_center, y_center = x+x_offset, y+y_offset
          patch_bbox = [x_center - l/2, y_center - l/2,
                        x_center + l/2, y_center + l/2]
          patch = crop_face(img, patch_bbox)
          # # debug
          # print patch.shape, scale, x_offset, y_offset, target[i, 0], target[i, 1]
          # patch = patch.copy()
          # patch_x, patch_y = patch_bbox[:2]
          # cv2.circle(patch, (int(x_center - patch_x), int(y_center - patch_y)), 1, (0, 255, 0), -1)
          # cv2.circle(patch, (int(x - patch_x), int(y - patch_y)), 1, (0, 0, 255), -1)
          # cv2.imshow('patch', patch)
          # cv2.waitKey(0)
          patch = cv2.resize(patch, (24, 24))
          data[:, :, (3*i):(3*i+3)] = patch
        data = data.transpose((2, 0, 1))  # 15x24x24, uint8
        target *= -1
        q_out.put(('data', [data, target]))


def lnet_writer_func(q_out, txt):
  file_counter = 0
  item_counter = 0
  fout_pattern = 'data/lnet_train_%03d.h5' if 'train' in txt else 'data/lnet_val_%03d.h5'
  fouts = []
  q = []

  def output_data(q, file_counter):
    file_counter += 1
    random.shuffle(q)
    n = len(q)
    data = np.zeros((n, 15, 24, 24), dtype=np.float32)
    target = np.zeros((n, 10), dtype=np.float32)
    for idx, (one_data, one_target) in enumerate(q):
      data[idx] = one_data
      target[idx] = one_target
    data = (data - 128.) / 128.  # process data
    fout = fout_pattern%file_counter
    fouts.append(fout)
    remove(fout)
    logger.info('write to %s', fout)
    with h5py.File(fout, 'w') as h5:
      h5['data'] = data
      h5['target'] = target
    return file_counter

  while True:
    stat, item = q_out.get()
    if stat == 'finish':
      file_counter = output_data(q, file_counter)
      q = []
      break
    item_counter += 1
    if item_counter%10000 == 0:
      logger.info('writes %d landmark data', item_counter)
    q.append(item)
    if len(q) >= cfg.DATASIZE_PER_H5:
      file_counter = output_data(q, file_counter)
      q = []

  remove(txt)
  with open(txt, 'w') as txt_out:
    for fout in fouts:
      txt_out.write('%s\n'%fout)
  logger.info("Finish")


def prepare(args):
  '''prepare data for lnet
  '''

  logger.info('loading CelebA')
  train_data, val_data = load_celeba()

  def gen(data, is_train):
    txt = 'data/lnet_train.txt' if is_train else 'data/lnet_val.txt'
    remove(txt)
    q_in = [mp.Queue() for i in range(cfg.WORKER_N)]
    q_out = mp.Queue(1024)
    fill_queues(data, q_in)
    readers = [mp.Process(target=lnet_reader_func, args=(q_in[i], q_out)) \
               for i in range(cfg.WORKER_N)]
    for p in readers:
      p.start()
    writer = mp.Process(target=lnet_writer_func, args=(q_out, txt))
    writer.start()
    for p in readers:
      p.join()
    q_out.put(('finish', []))
    writer.join()

  logger.info('writing train data')
  gen(train_data, True)
  logger.info('writing val data')
  gen(val_data, False)


# =========== train lnet ================

def train(args):
  '''train lnet using data prepare by `prepare()`
  '''

  def get_data_size(txt):
    size = 0
    with open(txt, 'r') as fin:
      for line in fin.readlines():
        line = line.strip()
        data = h5py.File(line, 'r')
        size += data['target'].shape[0]
        data.close()
    return size

  # init caffe
  np.random.seed(cfg.RNG_SEED)
  caffe.set_random_seed(cfg.RNG_SEED)
  if cfg.GPU_ID < 0:
    caffe.set_mode_cpu()
  else:
    caffe.set_mode_gpu()
    caffe.set_device(cfg.GPU_ID)
  # solver parameter setup
  batch_size = 128
  train_size = get_data_size('data/lnet_train.txt')
  val_size = get_data_size('data/lnet_val.txt')
  iter_train = train_size / batch_size
  iter_test = val_size / batch_size
  max_iter = args.epoch * iter_train
  final_model = 'tmp/lnet_iter_%d.caffemodel'%max_iter
  solver_param = caffe_pb2.SolverParameter()
  with open('proto/l_solver.prototxt', 'r') as fin:
    text_format.Merge(fin.read(), solver_param)
  solver_param.max_iter = max_iter
  solver_param.snapshot = iter_train
  solver_param.test_interval = iter_train
  solver_param.test_iter[0] = iter_test
  solver_param.base_lr = args.lr
  solver_param.gamma = args.lrw
  solver_param.stepsize = args.lrp * iter_train
  tmp_solver_prototxt = 'tmp/l_solver.prototxt'
  with open(tmp_solver_prototxt, 'w') as fout:
    fout.write(text_format.MessageToString(solver_param))
  # solver setup
  solver = caffe.SGDSolver(tmp_solver_prototxt)
  # train
  solver.solve(args.snapshot)
  shutil.copyfile(final_model, 'model/l.caffemodel');


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--prepare', action='store_true', help='prepare training data for lnet')
  parser.add_argument('--train', action='store_true', help='train lnet')
  parser.add_argument('--worker', type=int, default=8, help='workers to process the data')
  parser.add_argument('--gpu', type=int, default=0, help='gpu id to use, -1 for cpu')
  parser.add_argument('--epoch', type=int, default=20, help='train epoches')
  parser.add_argument('--snapshot', type=str, default=None, help='snapshot model')
  parser.add_argument('--lr', type=float, default=0.01, help='initial learning rate')
  parser.add_argument('--lrw', type=float, default=0.1, help='lr decay rate')
  parser.add_argument('--lrp', type=int, default=2, help='number of epoches to decay the lr')
  args = parser.parse_args()

  cfg.GPU_ID = args.gpu
  cfg.WORKER_N = args.worker

  print args

  if args.prepare:
    prepare(args)
  if args.train:
    train(args)
