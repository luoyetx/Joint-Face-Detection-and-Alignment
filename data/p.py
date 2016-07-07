#!/usr/bin/env python2.7
# coding = utf-8

import os
import shutil
import random
import argparse
import multiprocessing
import cv2
import lmdb
import numpy as np
from utils import load_wider, load_celeba
from utils import calc_IoU, calc_IoUs, check_bbox
from utils import draw_landmark, show_image
from utils import get_logger, get_face_size


READER_N = 4
RANDOM_FACE_N = 10
FACE_OVERLAP_THRESHOLD = 0.65
NONFACE_OVERLAP_THRESHOLD = 0.3
NONFACE_PER_IMAGE = 200

q_in = [multiprocessing.Queue() for i in range(READER_N)]
q_out = multiprocessing.Queue(1024)

logger = get_logger()
net_type = 'p'

G2 = 2*1024*1024*1024
G4 = 2*G2
G8 = 2*G4
G16 = 2*G8


def fill_queues(data, qs):
  data_n = len(data)
  queue_n = len(qs)
  for i in range(len(data)):
    qs[i%queue_n].put(data[i])


def remove_if_exists(db):
  if os.path.exists(db):
    shutil.rmtree(db)


# generate face part

def face_region_proposal(img, face_gt_bboxes, overlap_th):
  """generate bbox with overlap larger than overlap_th between any of bboxes
  :param img: image
  :param face_gt_bboxes: ground truth bboxes
  :param overlap_th: overlap threshold
  :return face_bboxes: proposal face bboxes
  :return face_offsets: offset between proposal face bboxes and ground truth bbox
  """
  face_bboxes = []
  face_offsets = []
  region = img.shape[:-1]

  def random_offset(bbox):
    x, y, w, h = bbox
    size = w
    sn = np.random.uniform(0.8, 1.3)
    xn = np.random.uniform(-0.3, 0.3)
    yn = np.random.uniform(-0.3, 0.3)
    # sn = np.random.choice([0.83, 0.91, 1, 1.1, 1.21])
    # xn = np.random.choice([-0.17, 0, 0.17])
    # yn = np.random.choice([-0.17, 0, 0.17])
    size = size*sn
    xn = x - xn*size
    yn = y - yn*size
    return [int(xn), int(yn), int(size), int(size)]

  def calc_offset(bbox, gt_bbox):
    dx = float(gt_bbox[0] - bbox[0]) / bbox[2]
    dy = float(gt_bbox[1] - bbox[1]) / bbox[3]
    ds = np.log(float(gt_bbox[2]) / bbox[2])
    return [dx, dy, ds]

  for gt_bbox in face_gt_bboxes:
    counter = 0
    tries = 0
    if check_bbox(gt_bbox, region):
      counter += 1
      face_bboxes.append(gt_bbox)
      face_offsets.append([0, 0, 1])
    while counter < RANDOM_FACE_N and tries < 50:
      tries += 1
      bbox = random_offset(gt_bbox)
      if check_bbox(bbox, region) and calc_IoU(bbox, gt_bbox) > overlap_th:
        counter += 1
        face_bboxes.append(bbox)
        face_offsets.append(calc_offset(bbox, gt_bbox))
        # print calc_offset(bbox, gt_bbox)
  return face_bboxes, face_offsets


def face_reader_func(q_in, q_out):
  while not q_in.empty():
    item = q_in.get()
    img_path, bboxes = item
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    bboxes, offsets = face_region_proposal(img, bboxes, FACE_OVERLAP_THRESHOLD)
    for bbox, offset in zip(bboxes, offsets):
      x, y, w, h = bbox
      face = img[y:y+h, x:x+w, :]
      face_size = get_face_size(net_type)
      face = cv2.resize(face, (face_size, face_size))
      face = face.transpose(2, 0, 1)
      face_data = face.tostring()  # uint8
      offset_data = np.asarray(offset, dtype=np.float32).tostring()
      q_out.put(('data', [face_data, offset_data]))
    # # for debug
    # for bbox in bboxes:
    #   x, y, w, h = bbox
    #   cv2.rectangle(img, (x, y), (x+w, y+h), (0,0,255), 1)
    # cv2.imshow('img', img)
    # cv2.waitKey(0)


def face_writer_func(q_out, db_name):
  map_size = G2
  db = lmdb.open(db_name, map_size=map_size)
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
    remove_if_exists(db_name)
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
  gen(train_data, '%snet_face_train'%net_type)
  logger.info('writing val data, %d images', len(val_data))
  gen(val_data, '%snet_face_val'%net_type)


# generate landmark face part

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
    face_size = get_face_size(net_type)
    face = cv2.resize(face, (face_size, face_size))
    face = face.transpose(2, 0, 1)
    face_data = face.tostring()  # string for lmdb, uint8
    landmark_data = landmark.tostring()  # float32
    q_out.put(('data', [face_data, landmark_data]))


def landmark_writer_func(q_out, db_name):
  map_size = G2
  db = lmdb.open(db_name, map_size=map_size)
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
    remove_if_exists(db_name)
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
  gen(train_data, '%snet_landmark_train'%net_type)
  logger.info('writing val data, %d images', len(val_data))
  gen(val_data, '%snet_landmark_val'%net_type)


# generate nonface part

def random_crop_nonface(img, face_bboxes, n):
    """random crop nonface region from img with size n
    :param img: image
    :param face_bboxes: face bboxes in this image
    :param n: number of nonface bboxes to crop
    :return nonface_bboxes: nonface region with size n
    """
    nonface_bboxes = []
    height, width = img.shape[:-1]
    range_x = width - 12
    range_y = height - 12
    while len(nonface_bboxes) < n:
      x, y = np.random.randint(range_x), np.random.randint(range_y)
      w = h = np.random.randint(low=12, high=min(width - x, height - y))
      nonface_bbox = [x, y, w, h]
      use_it = True
      for face_bbox in face_bboxes:
        if calc_IoU(nonface_bbox, face_bbox) > NONFACE_OVERLAP_THRESHOLD:
          use_it = False
          break
      if use_it:
        nonface_bboxes.append(nonface_bbox)
    return nonface_bboxes


def nonface_reader_func(q_in, q_out):
  while not q_in.empty():
    item = q_in.get()
    img_path, bboxes = item
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    bboxes = random_crop_nonface(img, bboxes, NONFACE_PER_IMAGE)
    for bbox in bboxes:
      x, y, w, h = bbox
      nonface = img[y:y+h, x:x+w, :]
      nonface_size = get_face_size(net_type)
      nonface = cv2.resize(nonface, (nonface_size, nonface_size))
      nonface = nonface.transpose(2, 0, 1)
      nonface_data = nonface.tostring()  # uint8
      q_out.put(('data', [nonface_data]))
    # # for debug
    # for bbox in bboxes:
    #   x, y, w, h = bbox
    #   cv2.rectangle(img, (x, y), (x+w, y+h), (0,0,255), 1)
    # cv2.imshow('img', img)
    # cv2.waitKey(0)


def nonface_writer_func(q_out, db_name):
  map_size = G4
  db = lmdb.open(db_name, map_size=map_size)  # 4G
  counter = 0
  with db.begin(write=True) as txn:
    while True:
      stat, item = q_out.get()
      if stat == 'finish':
        txn.put('size', str(counter))
        break
      face_data = item[0]
      face_key = '%08d_data'%counter
      txn.put(face_key, face_data)
      counter += 1
      if counter%1000 == 0:
        logger.info('process %d', counter)
  db.close()


def gen_nonface():
  """generate nonface data
  """
  logger.info('loading WIDER')
  train_data, val_data = load_wider()
  random.shuffle(train_data)
  random.shuffle(val_data)

  def gen(data, db_name):
    remove_if_exists(db_name)
    logger.info('fill queues')
    fill_queues(data, q_in)
    readers = [multiprocessing.Process(target=nonface_reader_func, args=(q_in[i], q_out)) \
               for i in range(READER_N)]
    for p in readers:
      p.start()
    writer = multiprocessing.Process(target=nonface_writer_func, args=(q_out, db_name))
    writer.start()
    for p in readers:
      p.join()
    q_out.put(('finish', []))
    writer.join()

  logger.info('writing train data, %d images', len(train_data))
  gen(train_data, '%snet_nonface_train'%net_type)
  logger.info('writing val data, %d images', len(val_data))
  gen(val_data, '%snet_nonface_val'%net_type)


def main(args):
  global net_type
  net_type = args.net
  if args.face:
    gen_face()
  if args.landmark:
    gen_landmark()
  if args.nonface:
    gen_nonface()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--net', type=str, default='p', help='net type')
  parser.add_argument('--face', action='store_true', help='generate face data')
  parser.add_argument('--landmark', action='store_true', help='generate landmark data')
  parser.add_argument('--nonface', action='store_true', help='generate nonface data')
  args = parser.parse_args()
  main(args)
