#!/usr/bin/env python2.7
# coding = utf-8

import os
import random
import argparse
import multiprocessing
import cv2
import lmdb
import numpy as np
from utils import load_wider, load_celeba
from utils import calc_IoU, calc_IoUs, check_bbox
from utils import draw_landmark, show_image
from utils import get_logger


READER_N = 2
RANDOM_OFFSET_N = 3
OVERLAP_THRESHOLD = 0.5

q_in = [multiprocessing.Queue() for i in range(READER_N)]
q_out = multiprocessing.Queue(1024)

logger = get_logger()


def fill_queues(data, qs):
  data_n = len(data)
  queue_n = len(qs)
  for i in range(len(data)):
    qs[i%queue_n].put(data[i])


def face_region_proposal(region, face_gt_bboxes, overlap_th):
  """generate bbox with overlap larger than overlap_th between any of bboxes
  :param region: [height, width]
  :param face_gt_bboxes: ground truth bboxes
  :param overlap_th: overlap threshold
  :return face_bboxes: proposal face bboxes
  :return face_offsets: offset between proposal face bboxes and ground truth bbox
  """
  face_bboxes = []
  face_offsets = []

  def random_offset(bbox, n):
    x, y, w, h = bbox
    x_center = x + w / 2
    y_center = y + h / 2
    size = (w + h) / 2
    x = x_center - size / 2
    y = y_center - size / 2
    sns = np.random.uniform(0.8, 1.3, n)
    xns = np.random.uniform(-0.2, 0.2, n)
    yns = np.random.uniform(-0.2, 0.2, n)
    sizes = size*sns
    xns = x - xns*sizes
    yns = y - yns*sizes
    for x, y, size in zip(xns, yns, sizes):
      yield [int(x), int(y), int(size), int(size)]

  def calc_offset(bbox, gt_bbox):
    dx = float(gt_bbox[0] - bbox[0]) / bbox[2]
    dy = float(gt_bbox[1] - bbox[1]) / bbox[3]
    dw = float(gt_bbox[2]) / bbox[2]
    dh = float(gt_bbox[3]) / bbox[3]
    return [dx, dy, dw, dh]

  for gt_bbox in face_gt_bboxes:
    for bbox in random_offset(gt_bbox, RANDOM_OFFSET_N):
      if check_bbox(bbox, region) and calc_IoU(bbox, gt_bbox) > overlap_th:
        face_bboxes.append(bbox)
        face_offsets.append(calc_offset(bbox, gt_bbox))
  return face_bboxes, face_offsets


def apply_offset(bbox, offset):
    """reverse, for debug
    """
    x, y, w, h = bbox
    dx, dy, dw, dh = offset
    w_ = w*dw
    h_ = h*dh
    x_ = x + dx*w_
    y_ = y + dy*h_
    return [int(x_), int(y_), int(w_), int(h_)]


def face_reader_func(q_in, q_out):
  while not q_in.empty():
    item = q_in.get()
    img_path, bboxes = item
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    region = img.shape[:-1]
    bboxes, offsets = face_region_proposal(region, bboxes, OVERLAP_THRESHOLD)
    for bbox, offset in zip(bboxes, offsets):
      x, y, w, h = bbox
      face = img[y:y+h, x:x+w, :]
      face = cv2.resize(face, (12, 12))
      face = face.transpose(2, 0, 1)
      face_data = face.tostring()  # uint8
      offset_data = np.asarray(offset, dtype=np.float32).tostring()
      q_out.put(('data', [face_data, offset_data]))
    #   # reverse
    #   rbbox = apply_offset(bbox, offset)
    #   x, y, w, h = rbbox
    #   cv2.rectangle(img, (x, y), (x+w, y+h), (0,0,255), 2)
    # cv2.imshow('img', img)
    # cv2.waitKey(0)


def face_writer_func(q_out, db_name):
  db = lmdb.open(db_name, map_size=2*1024*1024*1024)  # 2G
  counter = 0
  with db.begin(write=True) as txn:
    while True:
      stat, item = q_out.get()
      if stat == 'finish':
        txn.put('size', str(counter))
        break
      face_data, offset_data = item
      face_key = '%08d_data'%counter
      offset_key = '%08d_offset'%counter
      txn.put(face_key, face_data)
      txn.put(offset_key, offset_data)
      counter += 1
      if counter%1000 == 0:
        logger.info('process %d', counter)
  db.close()


def gen_face():
  """generate face data with bbox
  """
  logger.info('loading WIDER')
  train_data, val_data = load_wider()
  random.shuffle(train_data)
  random.shuffle(val_data)

  def gen(data, db_name):
    logger.info('fill queues')
    fill_queues(data, q_in)
    readers = [multiprocessing.Process(target=face_reader_func, args=(q_in[i], q_out)) \
               for i in range(READER_N)]
    for p in readers:
      p.start()
    writer = multiprocessing.Process(target=face_writer_func, args=(q_out, db_name))
    writer.start()
    for p in readers:
      p.join()
    q_out.put(('finish', []))
    writer.join()

  logger.info('writing train data, %d images', len(train_data))
  gen(train_data, 'pnet_face_train')
  logger.info('writing val data, %d images', len(val_data))
  gen(val_data, 'pnet_face_val')


def landmark_reader_func(q_in, q_out):
  while not q_in.empty():
    item = q_in.get()
    img_path, bbox, landmark = item
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
      logger.warning('read %s failed', img_path)
      continue
    region = img.shape[:-1]
    if not check_bbox(bbox, region):
      logger.warning('bbox out of range')
      continue
    x, y, w, h = bbox
    face = img[y:y+h, x:x+w, :]  # crop
    face = cv2.resize(face, (12, 12))
    face = face.transpose(2, 0, 1)
    face_data = face.tostring()  # string for lmdb, uint8
    landmark_data = landmark.tostring()  # float32
    q_out.put(('data', [face_data, landmark_data]))


def landmark_writer_func(q_out, db_name):
  db = lmdb.open(db_name, map_size=2*1024*1024*1024)  # 2G
  counter = 0
  with db.begin(write=True) as txn:
    while True:
      stat, item = q_out.get()
      if stat == 'finish':
        txn.put('size', str(counter))
        break
      face_data, landmark_data = item
      face_key = '%08d_data'%counter
      landmark_key = '%08d_landmark'%counter
      txn.put(face_key, face_data)
      txn.put(landmark_key, landmark_data)
      counter += 1
      if counter%1000 == 0:
        logger.info('process %d', counter)
  db.close()


def gen_landmark():
  """generate face data with landmark
  """
  logger.info('loading CelebA')
  data = load_celeba()
  logger.info('total images %d', len(data))
  random.shuffle(data)
  train_n = int(len(data)*0.8)
  train_data = data[:train_n]
  val_data = data[train_n:]

  def gen(data, db_name):
    logger.info('fill queues')
    fill_queues(data, q_in)
    readers = [multiprocessing.Process(target=landmark_reader_func, args=(q_in[i], q_out)) \
               for i in range(READER_N)]
    for p in readers:
      p.start()
    writer = multiprocessing.Process(target=landmark_writer_func, args=(q_out, db_name))
    writer.start()
    for p in readers:
      p.join()
    q_out.put(('finish', []))
    writer.join()

  logger.info('writing train data, %d images', len(train_data))
  gen(train_data, 'pnet_landmark_train')
  logger.info('writing val data, %d images', len(val_data))
  gen(val_data, 'pnet_landmark_val')


def main(args):
  if args.face:
    gen_face()
  if args.landmark:
    gen_landmark()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--face', action='store_true', help='generate face data')
  parser.add_argument('--landmark', action='store_true', help='generate landmark data')
  args = parser.parse_args()
  main(args)
