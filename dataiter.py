#!/usr/bin/env python2.7
# coding = utf-8

import multiprocessing
import cv2
import lmdb
import numpy as np
import mxnet as mx
from data.utils import load_wider
from data.utils import calc_IoU


NONFACE_VERLAP_THRESHOLD = 0.3
BATCH_QUEUE_LEN = 3
batch_queue = multiprocessing.Queue(BATCH_QUEUE_LEN)


class FaceDataGenerator(object):
  """Face Data Generator
  """

  def __init__(self, db_name, net_type='p', batch_size=256, shuffle=False):
    self.db_name = db_name
    self.db = lmdb.open(db_name)
    self.txn = self.db.begin()
    self.data_size = int(self.txn.get('size'))
    self.batch_size = batch_size
    self.epoch_size = self.data_size / self.batch_size
    self.shuffle = shuffle
    self.shuffle_idx = []
    self.reset()
    if net_type == 'p':
      self.face_shape = (3, 12, 12)
      self.face_data = np.zeros((batch_size, 3, 12, 12), dtype=np.float32)
    elif net_type == 'r':
      self.face_shape = (3, 24, 24)
      self.face_data = np.zeros((batch_size, 3, 24, 24), dtype=np.float32)
    else:
      assert net_type == 'o'
      self.face_shape = (3, 48, 48)
      self.face_data = np.zeros((batch_size, 3, 48, 48), dtype=np.float32)
    self.bbox_shape = 4
    self.bbox_data = np.zeros((batch_size, 4), dtype=np.float32)
    self.mask_data = np.zeros((batch_size, 2), dtype=np.float32)
    self.mask_data[:, 0] = 0  # no copy for bbox
    self.mask_data[:, 1] = 1  # need copy for landmark

  def __del__(self):
    self.txn.abort()
    self.db.close()

  def reset(self):
    self.current_batch_idx = 0
    if self.shuffle:
      self.shuffle_idx = np.random.permutation(self.data_size)
    else:
      self.shuffle_idx = np.arange(self.data_size)

  def __iter__(self):
    return self

  def __next__(self):
    return self.next()

  def next(self):
    if self.current_batch_idx >= self.epoch_size:
      raise StopIteration()
    offset = self.current_batch_idx * self.batch_size
    self.current_batch_idx += 1
    for i in range(self.batch_size):
      idx = offset + i
      db_idx = self.shuffle_idx[idx]
      face_key = '%08d_data'%db_idx
      bbox_key = '%08d_offset'%db_idx
      face_data = np.fromstring(self.txn.get(face_key), dtype=np.uint8).reshape(self.face_shape)
      bbox_data = np.fromstring(self.txn.get(bbox_key), dtype=np.float32).reshape(self.bbox_shape)
      self.face_data[i] = face_data
      self.bbox_data[i] = bbox_data
    # process face data
    self.face_data = (self.face_data - 128) / 128
    return (self.face_data, self.bbox_data, self.mask_data)


class LandmarkDataGenerator(object):
  """Landmark Data Generator
  """

  def __init__(self, db_name, net_type='p', batch_size=256, shuffle=False):
    self.db_name = db_name
    self.db = lmdb.open(db_name)
    self.txn = self.db.begin()
    self.data_size = int(self.txn.get('size'))
    self.batch_size = batch_size
    self.epoch_size = self.data_size / self.batch_size
    self.current_batch_idx = 0
    self.shuffle = shuffle
    self.shuffle_idx = []
    self.reset()
    if net_type == 'p':
      self.face_shape = (3, 12, 12)
      self.face_data = np.zeros((batch_size, 3, 12, 12), dtype=np.float32)
    elif net_type == 'r':
      self.face_shape = (3, 24, 24)
      self.face_data = np.zeros((batch_size, 3, 24, 24), dtype=np.float32)
    else:
      assert net_type == 'o'
      self.face_shape = (3, 48, 48)
      self.face_data = np.zeros((batch_size, 3, 48, 48), dtype=np.float32)
    self.landmark_shape = 10
    self.landmark_data = np.zeros((batch_size, 10), dtype=np.float32)
    self.mask_data = np.zeros((batch_size, 2), dtype=np.float32)
    self.mask_data[:, 0] = 1  # need copy for bbox
    self.mask_data[:, 1] = 0  # no copy for landmark

  def __del__(self):
    self.txn.abort()
    self.db.close()

  def reset(self):
    self.current_batch_idx = 0
    if self.shuffle:
      self.shuffle_idx = np.random.permutation(self.data_size)
    else:
      self.shuffle_idx = np.arange(self.data_size)

  def __iter__(self):
    return self

  def __next__(self):
    return self.next()

  def next(self):
    if self.current_batch_idx >= self.epoch_size:
      raise StopIteration()
    offset = self.current_batch_idx * self.batch_size
    self.current_batch_idx += 1
    for i in range(self.batch_size):
      idx = offset + i
      db_idx = self.shuffle_idx[idx]
      face_key = '%08d_data'%db_idx
      landmark_key = '%08d_landmark'%db_idx
      face_data = np.fromstring(self.txn.get(face_key), dtype=np.uint8).reshape(self.face_shape)
      landmark_data = np.fromstring(self.txn.get(landmark_key), dtype=np.float32).reshape(self.landmark_shape)
      self.face_data[i] = face_data
      self.landmark_data[i] = landmark_data
    # process face data
    self.face_data = (self.face_data - 128) / 128
    return (self.face_data, self.landmark_data, self.mask_data)


class NonFaceDataGenerator(object):
  """Nonface Data Generator
  """

  def __init__(self, bgs, net_type='p', batch_size=512, shuffle=False):
    """init
    :param bgs: bg list with bbox, [(img_path, bboxes), (img_path, bboxes)....]
    """
    self.bgs = bgs
    self.data_size = len(bgs)
    self.epoch_size = self.data_size
    self.batch_size = batch_size
    self.current_bg_idx = 0
    self.shuffle = shuffle
    self.shuffle_idx = []
    self.reset()
    if net_type == 'p':
      self.nonface_shape = (3, 12, 12)
      self.nonface_data = np.zeros((batch_size, 3, 12, 12), dtype=np.float32)
    elif net_type == 'r':
      self.nonface_shape = (3, 24, 24)
      self.nonface_data = np.zeros((batch_size, 3, 24, 24), dtype=np.float32)
    else:
      assert net_type == 'o'
      self.nonface_shape = (3, 48, 48)
      self.nonface_data = np.zeros((batch_size, 3, 48, 48), dtype=np.float32)
    self.mask_data = np.ones((batch_size, 2), dtype=np.float32)

  def reset(self):
    self.current_bg_idx = 0
    if self.shuffle:
      self.shuffle_idx = np.random.permutation(self.data_size)
    else:
      self.shuffle_idx = np.arange(self.data_size)

  def __iter__(self):
    return self

  def __next__(self):
    return self.next()

  def next(self):
    if self.current_bg_idx >= self.data_size:
      raise StopIteration()
    img_path, face_bboxes = self.bgs[self.shuffle_idx[self.current_bg_idx]]
    self.current_bg_idx += 1
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:  # read failed skip it
      return self.next()
    nonface_bboxes = self.random_crop_nonface(img, face_bboxes, self.batch_size)
    assert len(nonface_bboxes) == self.batch_size
    for idx, nonface_bbox in enumerate(nonface_bboxes):
      x, y, w, h = nonface_bbox
      nonface = img[y:y+h, x:x+w, :]
      nonface = cv2.resize(nonface, self.nonface_shape[1:])
      nonface = nonface.transpose((2, 0, 1))
      self.nonface_data[idx] = nonface
    # processing nonface data
    self.nonface_data = (self.nonface_data - 128) / 128
    return (self.nonface_data, self.mask_data)

  def random_crop_nonface(self, img, face_bboxes, n):
    """random crop nonface region from img with size n
    :param img: image
    :param face_bboxes: face bboxes in this image
    :param n: number of nonface bboxes to crop
    :return nonface_bboxes: nonface region with size n
    """
    nonface_bboxes = []
    height, width = img.shape[:-1]
    range_x = width - self.nonface_shape[2]
    range_y = height - self.nonface_shape[1]
    while len(nonface_bboxes) < n:
      x, y = np.random.randint(range_x), np.random.randint(range_y)
      w = h = np.random.randint(low=self.nonface_shape[1]/2, high=min(width - x, height - y))
      nonface_bbox = [x, y, w, h]
      use_it = True
      for face_bbox in face_bboxes:
        if calc_IoU(nonface_bbox, face_bbox) > NONFACE_VERLAP_THRESHOLD:
          use_it = False
          break
      if use_it:
        nonface_bboxes.append(nonface_bbox)
    return nonface_bboxes


class BatchGenerator(object):
  """generate batches from given dataset, put the batch in the queue for MTDataIter
  """

  def __init__(self, net_type='p', shuffle=False,
               face_db_name='data/pnet_face_train',
               landmark_db_name='data/pnet_landmark_train',
               nonface_bgs=[],
               face_batch_size=256,
               landmark_batch_size=256,
               nonface_batch_size=512):
    self.face_generator = FaceDataGenerator(db_name=face_db_name,
                                            net_type=net_type,
                                            batch_size=face_batch_size,
                                            shuffle=shuffle)
    self.landmark_generator = LandmarkDataGenerator(db_name=landmark_db_name,
                                                    net_type=net_type,
                                                    batch_size=landmark_batch_size,
                                                    shuffle=shuffle)
    self.nonface_generator = NonFaceDataGenerator(bgs=nonface_bgs,
                                                  net_type=net_type,
                                                  batch_size=nonface_batch_size,
                                                  shuffle=shuffle)
    self.face_batch_size = face_batch_size
    self.landmark_batch_size = landmark_batch_size
    self.nonface_batch_size = nonface_batch_size
    self.batch_size = face_batch_size + landmark_batch_size + nonface_batch_size
    if net_type == 'p':
      self.data_shape = (3, 12, 12)
      self.data = np.zeros((self.batch_size, 3, 12, 12), dtype=np.float32)
    elif net_type == 'r':
      self.data_shape = (3, 24, 24)
      self.data = np.zeros((self.batch_size, 3, 24, 24), dtype=np.float32)
    else:
      assert net_type == 'o'
      self.data_shape = (3, 48, 48)
      self.data = np.zeros((self.batch_size, 3, 48, 48), dtype=np.float32)
    self.bbox_shape = 4
    self.bbox_data = np.zeros((self.batch_size, 4), dtype=np.float32)
    self.landmark_shape = 10
    self.landmark_data = np.zeros((self.batch_size, 10), dtype=np.float32)
    self.mask_shape = 2
    self.mask_data = np.zeros((self.batch_size, 2), dtype=np.float32)

  def run(self):
    """generate one batch, data layout [face, landmark, nonface]
    :return data: data in a batch
    :return bbox_data: bbox in a batch
    :return landmark_data: landmark in a batch
    :return mask_data: mask in a batch
    """
    # face
    start, end = 0, self.face_batch_size
    face_data, bbox_data, mask_data = self.next_face_data().next()  # next() is needed for iter generator
    self.data[start:end] = face_data
    self.bbox_data[start:end] = bbox_data
    self.landmark_data[start:end].fill(0)  # must fill 0
    self.mask_data[start:end] = mask_data
    # landmark
    start, end = self.face_batch_size, self.face_batch_size+self.landmark_batch_size
    face_data, landmark_data, mask_data = self.next_landmark_data().next()
    self.data[start:end] = face_data
    self.bbox_data[start:end].fill(0)
    self.landmark_data[start:end] = landmark_data
    self.mask_data[start:end] = mask_data
    # nonface
    start, end = self.face_batch_size+self.landmark_batch_size, self.batch_size
    nonface_data, mask_data = self.next_nonface_data().next()
    self.data[start:end] = nonface_data
    self.bbox_data[start:end].fill(0)
    self.landmark_data[start:end].fill(0)
    self.mask_data[start:end] = mask_data
    return (self.data, self.bbox_data, self.landmark_data, self.mask_data)

  def next_face_data(self):
    """generate face data in a batch
    """
    while True:
      self.face_generator.reset()
      for batch in self.face_generator:
        yield batch  # face_data, bbox_data, mask_data

  def next_landmark_data(self):
    """generate landmark data in a batch
    """
    while True:
      self.landmark_generator.reset()
      for batch in self.landmark_generator:
        yield batch  # face_data, landmark_data, mask_data

  def next_nonface_data(self):
    """generate nonface data in a batch
    """
    while True:
      self.nonface_generator.reset()
      for batch in self.nonface_generator:
        yield batch  # nonface_data, mask_data


class MTDataIter(mx.io.DataIter):
  """Multi-task DataIter
  """

  def __init__(self, net_type='p', batch_sizes=[256, 256, 512]):
    """init the data iter
    :param net_type: 'p' or 'r' or 'o' for 3 nets
    :param batch_sizes: 3 batch_sizes for face, landmark, non-face
    """
    super(MTDataIter, self).__init__()
    self.net_type = net_type
    self.face_batch_size = batch_sizes[0]
    self.landmark_batch_size = batch_sizes[1]
    self.nonface_batch_size = batch_sizes[2]
    if net_type == 'p':
      pass


if __name__ == '__main__':
  # test
  face_generator = FaceDataGenerator('data/pnet_face_train', batch_size=2)
  face_data, bbox_data, mask_data = face_generator.next()
  print face_data, bbox_data, mask_data
  landmark_generator = LandmarkDataGenerator('data/pnet_landmark_train', batch_size=3)
  face_data, landmark_data, mask_data = landmark_generator.next()
  print face_data, landmark_data, mask_data
  train, val = load_wider()
  nonface_generator = NonFaceDataGenerator(train, batch_size=2, shuffle=True)
  nonface_data, mask_data = nonface_generator.next()
  print nonface_data, mask_data
  # test iteration
  for idx, batch in enumerate(face_generator):
    print idx
    if idx > 10: break
  # test BatchGenerator
  bgt = BatchGenerator(net_type='p', shuffle=True,
                       face_db_name='data/pnet_face_val',
                       landmark_db_name='data/pnet_landmark_val',
                       nonface_bgs=val,
                       face_batch_size=256,
                       landmark_batch_size=256,
                       nonface_batch_size=512)
  epoch_size = 100
  for i in range(epoch_size):
    print 'batch', i
    bgt.run()
