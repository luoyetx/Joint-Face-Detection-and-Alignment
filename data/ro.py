#!/usr/bin/env python2.7

import os
import argparse
import multiprocessing
import cv2
import lmdb
import numpy as np
from utils import load_wider, get_logger
from utils import calc_IoU, check_bbox, calc_offset, get_face_size


READER_N = 2
FACE_OVERLAP_THRESHOLD = 0.65
NONFACE_OVERLAP_THRESHOLD = 0.3
logger = get_logger()
net_type = 'r'

G2 = 2*1024*1024*1024
G4 = 2*G2
G8 = 2*G4
G12 = 3*G4
G16 = 2*G8
G24 = 2*G12
G32 = 2*G16
G48 = 2*G24


def load_wider_trained(net_type):
  train_txt, val_txt = 'tmp/wider_%s_train.txt'%net_type, 'tmp/wider_%s_val.txt'%net_type

  def load(text):
    fin = open(text, 'r')
    result = []
    while True:
      line = fin.readline()
      if not line: break  # eof
      img_path = line.strip()
      face_n = int(fin.readline().strip())

      bboxes = []
      for i in range(face_n):
        line = fin.readline().strip()
        components = line.split(' ')
        x, y, w, h = [float(_) for _ in components]
        bbox = [int(x), int(y), int(w), int(h)]
        bboxes.append(bbox)

      # # for debug
      # print len(bboxes)
      # img = cv2.imread(img_path)
      # for bbox in bboxes:
      #   x, y, w, h = bbox
      #   cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 1)
      # cv2.imshow('img', img)
      # cv2.waitKey(0)

      result.append([img_path, bboxes])
    fin.close()
    return result

  return load(train_txt), load(val_txt)


def reader(q_in, q_out_face, q_out_nonface):
  while not q_in.empty():
    item = q_in.get()
    img_path, gt_bboxes, bboxes = item
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    region = img.shape[:-1]
    face_bboxes = []
    nonface_bboxes = []
    for bbox in bboxes:
      bbox = [int(_) for _ in bbox]
      if check_bbox(bbox, region):
        face_flag = False  # not a face
        for gt_bbox in gt_bboxes:
          iou = calc_IoU(bbox, gt_bbox)
          if iou > FACE_OVERLAP_THRESHOLD:
            face_bboxes.append((bbox, calc_offset(bbox, gt_bbox)))
            break
        if not face_flag:
          nonface_bboxes.append(bbox)
    # # for debug
    # for bbox, offset in face_bboxes:
    #   x, y, w, h = bbox
    #   cv2.rectangle(img, (x, y), (x+w, y+h), (0,0,255))
    # cv2.imshow('face', img)
    # cv2.waitKey(0)

    nonface_size = face_size = get_face_size(net_type)
    # for bbox, offset in face_bboxes:
    #   x, y, w, h = bbox
    #   face = img[y:y+h, x:x+w, :]
    #   face = cv2.resize(face, (face_size, face_size))
    #   face = face.transpose(2, 0, 1)
    #   face_data = face.tostring()  # uint8
    #   offset_data = np.asarray(offset, dtype=np.float32).tostring()
    #   q_out_face.put(('data', [face_data, offset_data]))
    for bbox in nonface_bboxes:
      x, y, w, h = bbox
      nonface = img[y:y+h, x:x+w, :]
      nonface = cv2.resize(nonface, (nonface_size, nonface_size))
      nonface = nonface.transpose(2, 0, 1)
      nonface_data = nonface.tostring()  # uint8
      q_out_nonface.put(('data', [nonface_data]))


def face_writer(q_out_face, db_name):
  map_size = G8 if net_type == 'o' else G2
  db = lmdb.open(db_name, map_size=map_size)
  counter = 0
  with db.begin(write=True) as txn:
    while True:
      stat, item = q_out_face.get()
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


def nonface_writer(q_out_nonface, db_name):
  if net_type == 'r':
    map_size = G24
  else:
    map_size = G24
  db = lmdb.open(db_name, map_size=map_size)  # 4G
  counter = 0
  with db.begin(write=True) as txn:
    while True:
      stat, item = q_out_nonface.get()
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


def main(args):
  global net_type
  net_type = args.net
  logger.info('load ground truth wider')
  wider_gt_train, wider_gt_val = load_wider()
  logger.info('load wider')
  wider_train, wider_val = load_wider_trained(net_type)

  face_queue = multiprocessing.Queue(1024)
  nonface_queue = multiprocessing.Queue(1024)
  q_in = [multiprocessing.Queue() for i in range(READER_N)]

  def gen(data_gt, data, face_db_name, nonface_db_name):
    logger.info('gen %s'%face_db_name)
    assert len(data_gt) == len(data)
    n = len(data_gt)
    for i in range(n):
      img_path_gt, gt_bboxes = data_gt[i]
      img_path, bboxes = data[i]
      assert os.path.basename(img_path) == os.path.basename(img_path_gt)
      q_in[i%READER_N].put((img_path_gt, gt_bboxes, bboxes))
    readers = [multiprocessing.Process(target=reader, args=(q_in[i], face_queue, nonface_queue)) \
               for i in range(READER_N)]
    for p in readers:
      p.start()
    # writers = [multiprocessing.Process(target=face_writer, args=(face_queue, face_db_name)),
    #            multiprocessing.Process(target=nonface_writer, args=(nonface_queue, nonface_db_name))]
    writers = [multiprocessing.Process(target=nonface_writer, args=(nonface_queue, nonface_db_name))]
    for p in writers:
      p.start()
    for p in readers:
      p.join()
    face_queue.put(('finish', []))
    nonface_queue.put(('finish', []))
    for p in writers:
      p.join()

  gen(wider_gt_train, wider_train, 'data/%snet_face_train'%net_type, 'data/%snet_nonface_train'%net_type)
  gen(wider_gt_val, wider_val, 'data/%snet_face_val'%net_type, 'data/%snet_nonface_val'%net_type)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--net', type=str, default='p', help='net type')
  args = parser.parse_args()
  assert args.net in  ['r', 'o']
  main(args)
