#!/usr/bin/env python2.7

import os
import shutil
import random
import logging
import argparse
import multiprocessing
import cv2
import lmdb
import caffe
import numpy as np
from jfda.config import cfg
from jfda.utils import load_wider, load_celeba
from jfda.utils import get_logger, crop_face
from jfda.detector import JfdaDetector

import pyximport
pyximport.install(setup_args={'include_dirs': np.get_include()})
from bbox import bbox_overlaps


logger = get_logger()

G8 = 8*1024*1024*1024
G16 = 2*G8
G24 = 3*G8
G32 = 4*G8


def fill_queues(data, qs):
  data_n = len(data)
  queue_n = len(qs)
  for i in range(len(data)):
    qs[i%queue_n].put(data[i])

def remove_if_exists(db):
  if os.path.exists(db):
    logger.info('remove %s'%db)
    shutil.rmtree(db)

def get_detector():
  nets = cfg.PROPOSAL_NETS[cfg.NET_TYPE]
  if nets is None or not cfg.USE_DETECT:
    detector = None
  else:
    if cfg.GPU_ID >= 0:
      caffe.set_mode_gpu()
      caffe.set_device(cfg.GPU_ID)
    else:
      caffe.set_mode_cpu()
    detector = JfdaDetector(nets)
  return detector


# =========== region proposal =============================

def sliding_windows(x, y, width, height, kw, kh, sw, sh):
  '''given a region (x, y, width, height), return sliding window locations (x1, y1, x2, y2)
  x, y: region top left position
  width, height: region width and height
  kw, kh: window width and height
  sw, sh: stride width and height
  '''
  xs = np.arange(0, width-kw, sw)
  ys = np.arange(0, height-kh, sh)
  xs, ys = np.meshgrid(xs, ys)
  xy = np.vstack([xs.ravel(), ys.ravel()]).transpose()
  wh = np.array([kw, kh])
  bbox = np.hstack([xy, np.tile(wh, (len(xy), 1))])
  bbox[:, 0] += x
  bbox[:, 1] += y
  bbox[:, 2] += bbox[:, 0]
  bbox[:, 3] += bbox[:, 1]
  return bbox.astype(np.float32)


def proposal(img, gt_bboxes, detector=None):
  '''given an image with face bboxes, proposal negatives, positives and part faces
  for rNet and oNet, we use previous networks to proposal bboxes
  Return
    (negatives, positives, part)
    negatives: [data, bbox]
    positives: [(data, bbox, bbox_target)]
    part: [(data, bbox, bbox_target)]
  '''
  # ======================= proposal for rnet and onet ==============
  if detector is not None:
    bboxes = detector.detect(img, **cfg.DETECT_PARAMS)
    # # maybe sort it by score in descending order
    # bboxes = bboxes[bboxes[:, 4].argsort()[::-1]]
    # keep bbox info, drop score, offset and landmark
    bboxes = bboxes[:, :4]
    ovs = bbox_overlaps(bboxes, gt_bboxes)
    ovs_max = ovs.max(axis=1)
    ovs_idx = ovs.argmax(axis=1)
    pos_idx = np.where(ovs_max > cfg.FACE_OVERLAP)[0]
    neg_idx = np.where(ovs_max < cfg.NONFACE_OVERLAP)[0]
    part_idx = np.where(np.logical_and(ovs_max > cfg.PARTFACE_OVERLAP, ovs_max <= cfg.FACE_OVERLAP))[0]
    # pos
    positives = []
    for idx in pos_idx:
      bbox = bboxes[idx].reshape(4)
      gt_bbox = gt_bboxes[ovs_idx[idx]]
      data = crop_face(img, bbox)
      if data is None:
        continue
      # cv2.imshow('pos', data)
      # cv2.waitKey()
      k = bbox[2] - bbox[0]
      bbox_target = (gt_bbox - bbox) / k
      positives.append((data, bbox, bbox_target))
    # part
    part = []
    for idx in part_idx:
      bbox = bboxes[idx].reshape(4)
      gt_bbox = gt_bboxes[ovs_idx[idx]]
      data = crop_face(img, bbox)
      if data is None:
        continue
      # cv2.imshow('part', data)
      # cv2.waitKey()
      k = bbox[2] - bbox[0]
      bbox_target = (gt_bbox - bbox) / k
      part.append((data, bbox, bbox_target))
    # neg
    negatives = []
    np.random.shuffle(neg_idx)
    for idx in neg_idx[:cfg.NEG_PER_IMAGE]:
      bbox = bboxes[idx].reshape(4)
      data = crop_face(img, bbox)
      if data is None:
        continue
      # cv2.imshow('neg', data)
      # cv2.waitKey()
      negatives.append((data, bbox))
    return negatives, positives, part

  # ======================= proposal for pnet =======================
  height, width = img.shape[:-1]
  negatives, positives, part = [], [], []

  # ===== proposal positives =====
  for gt_bbox in gt_bboxes:
    x, y = gt_bbox[:2]
    w, h = gt_bbox[2]-gt_bbox[0], gt_bbox[3]-gt_bbox[1]
    this_positives = []
    for scale in cfg.POS_PROPOSAL_SCALES:
      k = max(w, h) * scale
      stride = cfg.POS_PROPOSAL_STRIDE
      s = k * stride
      offset_x = (0.5 + np.random.rand()) * k / 2.
      offset_y = (0.5 + np.random.rand()) * k / 2.
      candidates = sliding_windows(x-offset_x, y-offset_y, w+2*offset_x, h+2*offset_y, k, k, s, s)
      ovs = bbox_overlaps(candidates, gt_bbox.reshape((1, 4)))
      ovs = ovs.reshape((1, len(candidates)))[0]
      pos_bboxes = candidates[ovs > cfg.FACE_OVERLAP, :]
      if len(pos_bboxes) > 0:
        np.random.shuffle(pos_bboxes)
      for bbox in pos_bboxes[:cfg.POS_PER_FACE]:
        data = crop_face(img, bbox)
        if data is None:
          continue
        # cv2.imshow('positive', data)
        # cv2.waitKey()
        bbox_target = (gt_bbox - bbox) / k
        this_positives.append((data, bbox, bbox_target))
    random.shuffle(this_positives)
    positives.extend(this_positives[:cfg.POS_PER_FACE])

  # ===== proposal part faces =====
  for gt_bbox in gt_bboxes:
    x, y = gt_bbox[:2]
    w, h = gt_bbox[2]-gt_bbox[0], gt_bbox[3]-gt_bbox[1]
    this_part = []
    for scale in cfg.PART_PROPOSAL_SCALES:
      k = max(w, h) * scale
      stride = cfg.PART_PROPOSAL_STRIDE
      s = k * stride
      offset_x = (0.5 + np.random.rand()) * k / 2.
      offset_y = (0.5 + np.random.rand()) * k / 2.
      candidates = sliding_windows(x-offset_x, y-offset_y, w+2*offset_x, h+2*offset_y, k, k, s, s)
      ovs = bbox_overlaps(candidates, gt_bbox.reshape((1, 4)))
      ovs = ovs.reshape((1, len(candidates)))[0]
      part_bboxes = candidates[np.logical_and(ovs > cfg.PARTFACE_OVERLAP, ovs <= cfg.FACE_OVERLAP), :]
      if len(part_bboxes) > 0:
        np.random.shuffle(part_bboxes)
      for bbox in part_bboxes[:cfg.PART_PER_FACE]:
        data = crop_face(img, bbox)
        if data is None:
          continue
        # cv2.imshow('part', data)
        # cv2.waitKey()
        bbox_target = (gt_bbox - bbox) / k
        this_part.append((data, bbox, bbox_target))
    random.shuffle(this_part)
    part.extend(this_part[:cfg.POS_PER_FACE])

  # ===== proposal negatives =====
  for gt_bbox in gt_bboxes:
    x, y = gt_bbox[:2]
    w, h = gt_bbox[2]-gt_bbox[0], gt_bbox[3]-gt_bbox[1]
    this_negatives = []
    for scale in cfg.NEG_PROPOSAL_SCALES:
      k = max(w, h) * scale
      stride = cfg.NEG_PROPOSAL_STRIDE
      s = k * stride
      offset_x = (0.5 + np.random.rand()) * k / 2.
      offset_y = (0.5 + np.random.rand()) * k / 2.
      candidates = sliding_windows(x-offset_x, y-offset_y, w+2*offset_x, h+2*offset_y, k, k, s, s)
      ovs = bbox_overlaps(candidates, gt_bboxes)
      neg_bboxes = candidates[ovs.max(axis=1) < cfg.NONFACE_OVERLAP, :]
      if len(neg_bboxes) > 0:
        np.random.shuffle(neg_bboxes)
      for bbox in neg_bboxes[:cfg.NEG_PER_FACE]:
        data = crop_face(img, bbox)
        if data is None:
          continue
        # cv2.imshow('negative', data)
        # cv2.waitKey()
        this_negatives.append((data, bbox))
    random.shuffle(this_negatives)
    negatives.extend(this_negatives[:cfg.NEG_PER_FACE])

  # negatives from global image random crop
  max_num_from_fr = int(cfg.NEG_PER_IMAGE * cfg.NEG_FROM_FR_RATIO)
  if len(negatives) > max_num_from_fr:
    random.shuffle(negatives)
    negatives = negatives[:max_num_from_fr]
  bbox_neg = []
  range_x, range_y = width - cfg.NEG_MIN_SIZE, height - cfg.NEG_MIN_SIZE
  for i in xrange(cfg.NEG_PROPOSAL_RATIO * cfg.NEG_PER_IMAGE):
    x1, y1 = np.random.randint(range_x), np.random.randint(range_y)
    w = h = np.random.randint(low=cfg.NEG_MIN_SIZE, high=min(width-x1, height-y1))
    x2, y2 = x1 + w, y1 + h
    bbox_neg.append([x1, y1, x2, y2])
    if x2 > width or y2 > height:
      print 'hhhh'
  bbox_neg = np.asarray(bbox_neg, dtype=gt_bboxes.dtype)
  ovs = bbox_overlaps(bbox_neg, gt_bboxes)
  bbox_neg = bbox_neg[ovs.max(axis=1) < cfg.NONFACE_OVERLAP]
  np.random.shuffle(bbox_neg)
  remain = cfg.NEG_PER_IMAGE - len(negatives)
  bbox_neg = bbox_neg[:remain]

  # for bbox in bbox_neg:
  #   x1, y1, x2, y2 = bbox
  #   x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
  #   cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
  # cv2.imshow('neg', img)
  # cv2.waitKey()

  for bbox in bbox_neg:
    data = crop_face(img, bbox)
    negatives.append((data, bbox))
  return negatives, positives, part


# =========== WIDER ================

def gen_wider():
  logger.info('loading WIDER')
  train_data, val_data = load_wider()
  logger.info('total images, train: %d, val: %d', len(train_data), len(val_data))
  train_faces = reduce(lambda acc, x: acc + len(x[1]), train_data, 0)
  val_faces = reduce(lambda acc, x: acc + len(x[1]), val_data, 0)
  logger.info('total faces, train: %d, val: %d', train_faces, val_faces)

  def gen(data, db_names):
    for db_name in db_names: remove_if_exists(db_name)
    logger.info('fill queues')
    q_in = [multiprocessing.Queue() for i in range(cfg.WORKER_N)]
    q_out = multiprocessing.Queue(1024)
    fill_queues(data, q_in)
    readers = [multiprocessing.Process(target=wider_reader_func, args=(q_in[i], q_out)) \
               for i in range(cfg.WORKER_N)]
    for p in readers:
      p.start()
    writer = multiprocessing.Process(target=wider_writer_func, args=(q_out, db_names))
    writer.start()
    for p in readers:
      p.join()
    q_out.put(('finish', []))
    writer.join()

  logger.info('writing train data, %d images', len(train_data))
  db_names = ['data/%snet_positive_train'%cfg.NET_TYPE,
              'data/%snet_negative_train'%cfg.NET_TYPE,
              'data/%snet_part_train'%cfg.NET_TYPE]
  gen(train_data, db_names)
  logger.info('writing val data, %d images', len(val_data))
  db_names = ['data/%snet_positive_val'%cfg.NET_TYPE,
              'data/%snet_negative_val'%cfg.NET_TYPE,
              'data/%snet_part_val'%cfg.NET_TYPE]
  gen(val_data, db_names)


def wider_reader_func(q_in, q_out):
  input_size = cfg.NET_INPUT_SIZE[cfg.NET_TYPE]
  detector = get_detector()
  counter = 0
  while not q_in.empty():
    item = q_in.get()
    counter += 1
    if counter % 1000 == 0:
      logger.info('%s reads %d', multiprocessing.current_process().name, counter)
    img_path, bboxes = item
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
      logger.warning('read %s failed', img_path)
      continue
    negatives, positives, part = proposal(img, bboxes, detector)

    for data, _ in negatives:
      data = cv2.resize(data, (input_size, input_size))
      data = data.tostring()  # string for lmdb, uint8
      q_out.put(('negative', [data]))
    for data, _, bbox_target in positives:
      data = cv2.resize(data, (input_size, input_size))
      data = data.tostring()  # string for lmdb, uint8
      bbox_target = bbox_target.astype(np.float32).tostring()  # float32
      q_out.put(('positive', [data, bbox_target]))
    for data, _, bbox_target in part:
      data = cv2.resize(data, (input_size, input_size))
      data = data.tostring()  # string for lmdb, uint8
      bbox_target = bbox_target.astype(np.float32).tostring()  # float32
      q_out.put(('part', [data, bbox_target]))


def wider_writer_func(q_out, db_names):
  db_pos = lmdb.open(db_names[0], map_size=G16)
  db_neg = lmdb.open(db_names[1], map_size=G16)
  db_part = lmdb.open(db_names[2], map_size=G16)
  txn_pos = db_pos.begin(write=True)
  txn_neg = db_neg.begin(write=True)
  txn_part = db_part.begin(write=True)

  idx_pos, idx_neg, idx_part = 0, 0, 0
  q_pos, q_neg, q_part = [], [], []

  def fill(txn, items, idx, has_bbox=True):
    random.shuffle(items)
    for item in items:
      data_key = '%08d_data'%idx
      txn.put(data_key, item[0])
      if has_bbox:
        bbox_key = '%08d_bbox'%idx
        txn.put(bbox_key, item[1])
      idx += 1
    return idx

  counter = 0
  pos_counter, neg_counter, part_counter = 0, 0, 0
  while True:
    stat, item = q_out.get()
    counter += 1
    if counter % 10000 == 0:
      logger.info('writes %d positives, %d negatives, %d part', pos_counter, neg_counter, part_counter)
    if stat == 'positive':
      pos_counter += 1
      q_pos.append(item)
      if len(q_pos) >= cfg.SHUFFLE_SIZE:
        idx_pos = fill(txn_pos, q_pos, idx_pos, True)
        q_pos = []
    elif stat == 'negative':
      neg_counter += 1
      q_neg.append(item)
      if len(q_neg) >= cfg.SHUFFLE_SIZE:
        idx_neg = fill(txn_neg, q_neg, idx_neg, False)
        q_neg = []
    elif stat == 'part':
      part_counter += 1
      q_part.append(item)
      if len(q_part) >= cfg.SHUFFLE_SIZE:
        idx_part = fill(txn_part, q_part, idx_part, True)
        q_part = []
    else:
      # stat == 'finish'
      idx_pos = fill(txn_pos, q_pos, idx_pos, True)
      txn_pos.put('size', str(idx_pos))
      idx_neg = fill(txn_neg, q_neg, idx_neg, False)
      txn_neg.put('size', str(idx_neg))
      idx_pos = fill(txn_part, q_part, idx_part, True)
      txn_part.put('size', str(idx_part))
      break

  txn_pos.commit()
  txn_neg.commit()
  txn_part.commit()
  db_pos.close()
  db_neg.close()
  db_part.close()
  logger.info('Finish')


# =========== CelebA ===============

def gen_celeba():
  logger.info('loading CelebA')
  train_data, val_data = load_celeba()
  logger.info('total images, train: %d, val: %d', len(train_data), len(val_data))

  def gen(data, db_name):
    remove_if_exists(db_name)
    logger.info('fill queues')
    q_in = [multiprocessing.Queue() for i in range(cfg.WORKER_N)]
    q_out = multiprocessing.Queue(1024)
    fill_queues(data, q_in)
    readers = [multiprocessing.Process(target=celeba_reader_func, args=(q_in[i], q_out)) \
               for i in range(cfg.WORKER_N)]
    for p in readers:
      p.start()
    writer = multiprocessing.Process(target=celeba_writer_func, args=(q_out, db_name))
    writer.start()
    for p in readers:
      p.join()
    q_out.put(('finish', []))
    writer.join()

  logger.info('writing train data, %d images', len(train_data))
  gen(train_data, 'data/%snet_landmark_train'%cfg.NET_TYPE)
  logger.info('writing val data, %d images', len(val_data))
  gen(val_data, 'data/%snet_landmark_val'%cfg.NET_TYPE)


def celeba_reader_func(q_in, q_out):

  def vertify_bbox(bbox, landmark):
    return True

  input_size = cfg.NET_INPUT_SIZE[cfg.NET_TYPE]
  detector = get_detector()
  counter = 0
  while not q_in.empty():
    item = q_in.get()
    counter += 1
    if counter%1000 == 0:
      logger.info('%s reads %d', multiprocessing.current_process().name, counter)
    img_path, bbox, landmark = item
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
      logger.warning('read %s failed', img_path)
      continue
    bbox = np.asarray(bbox, dtype=np.float32).reshape((1, -1))
    _1, bboxes, _2 = proposal(img, bbox, detector)
    np.random.shuffle(bboxes)
    for data, bbox, _ in bboxes[:cfg.LANDMARK_PER_FACE]:
      # make sure landmark points are in bbox
      landmark1 = landmark.reshape((-1, 2)).copy()
      if not vertify_bbox(bbox, landmark1):
        continue
      # # debug
      # img1 = img.copy()
      # x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
      # cv2.rectangle(img1, (x1, y1), (x2, y2), (0, 0, 255), 2)
      # for x, y in landmark1:
      #   x, y = int(x), int(y)
      #   cv2.circle(img1, (x, y), 2, (0, 255, 0), -1)
      # cv2.imshow('landmark', img1)
      # cv2.waitKey(0)
      # normalize landmark
      w, h = bbox[2]-bbox[0], bbox[3]-bbox[1]
      landmark1[:, 0] = (landmark1[:, 0] - bbox[0]) / w
      landmark1[:, 1] = (landmark1[:, 1] - bbox[1]) / h
      landmark1 = landmark1.reshape(-1)
      # format data
      data = cv2.resize(data, (input_size, input_size))
      data = data.tostring()  # string for lmdb, uint8
      landmark1 = landmark1.astype(np.float32).tostring()  # float32
      q_out.put(('data', [data, landmark1]))


def celeba_writer_func(q_out, db_name):
  map_size = G16
  db = lmdb.open(db_name, map_size=map_size)
  counter = 0
  with db.begin(write=True) as txn:
    while True:
      stat, item = q_out.get()
      if stat == 'finish':
        txn.put('size', str(counter))
        break
      data, landmark = item
      data_key = '%08d_data'%counter
      landmark_key = '%08d_landmark'%counter
      txn.put(data_key, data)
      txn.put(landmark_key, landmark)
      counter += 1
      if counter%1000 == 0:
        logger.info('writes %d landmark faces', counter)
  db.close()
  logger.info('Finish')


def test():
  os.system('rm -rf tmp/pos/*')
  os.system('rm -rf tmp/neg/*')
  os.system('rm -rf tmp/part/*')
  logger.info('Load WIDER')
  train_data, val_data = load_wider()
  img_path, bboxes = train_data[np.random.choice(len(train_data))]
  bboxes = np.asarray(bboxes)
  img = cv2.imread(img_path, cv2.IMREAD_COLOR)
  detector = JfdaDetector(cfg.PROPOSAL_NETS['r'])
  negatives, positives, part = proposal(img, bboxes, detector)
  logger.info('%d gt_bboxes', len(bboxes))
  logger.info('%d negatives, %d positives, %d part', len(negatives), len(positives), len(part))
  for i, (data, bbox_target) in enumerate(positives):
    cv2.imwrite('tmp/pos/%03d.jpg'%i, data)
  for i, (data) in enumerate(negatives):
    cv2.imwrite('tmp/neg/%03d.jpg'%i, data)
  for i, (data, bbox_target) in enumerate(part):
    cv2.imwrite('tmp/part/%03d.jpg'%i, data)
  cv2.imwrite('tmp/test.jpg', img)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--net', type=str, default='p', help='net type')
  parser.add_argument('--celeba', action='store_true', help='generate face data')
  parser.add_argument('--wider', action='store_true', help='generate landmark data')
  parser.add_argument('--gpu', type=int, default=0, help='gpu device')
  parser.add_argument('--detect', action='store_true', help='use previous network detection')
  parser.add_argument('--worker', type=int, default=8, help='workers to process the data')
  parser.add_argument('--test', action='store_true', help='just simple test')
  args = parser.parse_args()

  cfg.GPU_ID = args.gpu
  cfg.NET_TYPE = args.net
  cfg.USE_DETECT = args.detect
  cfg.WORKER_N = args.worker

  if args.test:
    test()
  if args.wider:
    gen_wider()
  if args.celeba:
    gen_celeba()
