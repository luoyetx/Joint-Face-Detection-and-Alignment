#!/usr/bin/env python2.7
# coding = utf-8
#
# generate batch for the network.
# There're three type of data
#  1. face with bbox offset
#  2. face with landmark
#  3. non-face
# We generate this data in a batch with face_batch_size, landmark_batch_size and nonface_batch_size.
#  1. data, this is face data resized to 3 x ? x ?, where ? is 12 or 24 or 48
#  2. bbox, bbox offset, not all data type has this field
#  3. landmark, 5 landmark points, not all data type has this field
#  4. mask, this data field indicate the data point has bbox or landmark.
#     1. mask[0] for bbox offset and mask[1] for landmark
#     2. face with bbox offset, mask[0] = 0, mask[1] = 1, landmark data need to copy from network output
#     3. face with landmark offset, mask[0] = 1, mask[1] = 0, bbox offset data need to copy from network output
#     4. nonface, mask[0] = mask[1] = 1, both need to copy from network output

import multiprocessing
import cv2
import lmdb
import numpy as np
from .utils import calc_IoU, get_face_size


def preprocess_face_data(face_data):
  """preprocess face data before feeding to network, original data lie in [0, 255]
  """
  res = (face_data - 128) / 128
  return res


class BaseGenerator(object):
  """Base Generator
  """

  def __init__(self, db_name, net_type, batch_size):
    self.db_name = db_name
    self.net_type = net_type
    self.batch_size = batch_size
    # load data
    db = lmdb.open(db_name)
    with db.begin() as txn:
      self.data_size = self.get_size(txn)
      self.epoch_size = self.data_size / self.batch_size
      self.load_full_data(txn)
    db.close()
    # reset status
    self.reset()

  def reset(self):
    self.batch_start_idx = 0
    #self.shuffle_idx = np.random.permutation(self.data_size)
    self.shuffle_idx = np.arange(self.data_size)

  def load_full_data(self, txn):
    """subclass need to impl, load full data
    txn: lmdb context
    """
    pass

  def get_mini_batch(self, idx):
    """subclass need to impl, generate mini-batch, raise StopIteration if end of datasize
    idx: data index
    return: (data-1, data-2, ...)
    """
    pass

  def get_size(self, txn):
    """subclass need to impl, get data size
    txn: lmdb context
    return: full data size
    """
    size = int(txn.get('size'))
    return size

  def __iter__(self):
    return self

  def __next__(self):
    self.next()

  def next(self):
    """subclass need to impl, get mini-batch
    return: mini-batch
    """
    start = self.batch_start_idx
    end = start + self.batch_size
    self.batch_start_idx = end
    if end > self.data_size:
      raise StopIteration()
    idx = self.shuffle_idx[start:end]
    mini_batch = self.get_mini_batch(idx)
    return mini_batch


class FaceDataGenerator(BaseGenerator):
  """Face Data Generator
  given lmdb file path, generate a batch with size = batch_size each time
  """

  def __init__(self, db_name, net_type='p', batch_size=256):
    super(FaceDataGenerator, self).__init__(db_name, net_type, batch_size)

  def load_full_data(self, txn):
    face_size = get_face_size(self.net_type)
    bbox_size = 3
    face_shape = (3, face_size, face_size)
    bbox_shape = (bbox_size,)
    self.face_data = np.zeros((self.data_size, 3, face_size, face_size), dtype=np.uint8)
    self.bbox_data = np.zeros((self.data_size, bbox_size), dtype=np.float32)
    for i in xrange(self.data_size):
      face_key = '%08d_data'%i
      bbox_key = '%08d_offset'%i
      self.face_data[i] = np.fromstring(txn.get(face_key), dtype=np.uint8).reshape(face_shape)
      self.bbox_data[i] = np.fromstring(txn.get(bbox_key), dtype=np.float32).reshape(bbox_shape)

  def get_mini_batch(self, idx):
    face_data = self.face_data[idx].copy().astype(np.float32)
    bbox_data = self.bbox_data[idx].copy()
    face_data = preprocess_face_data(face_data)
    return (face_data, bbox_data)


class LandmarkDataGenerator(BaseGenerator):
  """Landmark Data Generator
  """

  def __init__(self, db_name, net_type='p', batch_size=256):
    super(LandmarkDataGenerator, self).__init__(db_name, net_type, batch_size)

  def load_full_data(self, txn):
    face_size = get_face_size(self.net_type)
    landmark_size = 10
    face_shape = (3, face_size, face_size)
    landmark_shape = (landmark_size,)
    self.face_data = np.zeros((self.data_size, 3, face_size, face_size), dtype=np.uint8)
    self.landmark_data = np.zeros((self.data_size, landmark_size), dtype=np.float32)
    for i in xrange(self.data_size):
      face_key = '%08d_data'%i
      landmark_key = '%08d_landmark'%i
      self.face_data[i] = np.fromstring(txn.get(face_key), dtype=np.uint8).reshape(face_shape)
      self.landmark_data[i] = np.fromstring(txn.get(landmark_key), dtype=np.float32).reshape(landmark_shape)

  def get_mini_batch(self, idx):
    face_data = self.face_data[idx].copy().astype(np.float32)
    landmark_data = self.landmark_data[idx].copy()
    face_data = preprocess_face_data(face_data)
    return (face_data, landmark_data)


class NonFaceDataGenerator(BaseGenerator):
  """Nonface Data Generator
  """

  def __init__(self, db_name, net_type='p', batch_size=512):
    super(NonFaceDataGenerator, self).__init__(db_name, net_type, batch_size)

  def load_full_data(self, txn):
    nonface_size = get_face_size(self.net_type)
    nonface_shape = (3, nonface_size, nonface_size)
    self.nonface_data = np.zeros((self.data_size, 3, nonface_size, nonface_size), dtype=np.uint8)
    for i in xrange(self.data_size):
      nonface_key = '%08d_data'%i
      self.nonface_data[i] = np.fromstring(txn.get(nonface_key), dtype=np.uint8).reshape(nonface_shape)

  def get_mini_batch(self, idx):
    nonface_data = self.nonface_data[idx].copy().astype(np.float32)
    nonface_data = preprocess_face_data(nonface_data)
    return (nonface_data)


class BatchGenerator(multiprocessing.Process):
  """generate batches from given dataset, put the batch in the queue for MTDataIter
  """

  def __init__(self, queue, net_type='p',
               face_db_name='data/pnet_face_train',
               landmark_db_name='data/pnet_landmark_train',
               nonface_db_name='data/pnet_nonface_train',
               face_batch_size=126,
               landmark_batch_size=126,
               nonface_batch_size=512):
    super(BatchGenerator, self).__init__()
    self.queue = queue
    self.face_generator = FaceDataGenerator(db_name=face_db_name,
                                            net_type=net_type,
                                            batch_size=face_batch_size)
    self.landmark_generator = LandmarkDataGenerator(db_name=landmark_db_name,
                                                    net_type=net_type,
                                                    batch_size=landmark_batch_size)
    self.nonface_generator = NonFaceDataGenerator(db_name=nonface_db_name,
                                                  net_type=net_type,
                                                  batch_size=nonface_batch_size)
    self.face_batch_size = face_batch_size
    self.landmark_batch_size = landmark_batch_size
    self.nonface_batch_size = nonface_batch_size
    self.batch_size = face_batch_size + landmark_batch_size + nonface_batch_size
    # face data
    size = get_face_size(net_type)
    self.data_shape = (3, size, size)
    self.data = np.zeros((self.batch_size, 3, size, size), dtype=np.float32)
    # face label
    self.label_shape = (1,)
    self.label_data = np.zeros((self.batch_size,), dtype=np.float32)
    # face bbox offset
    self.bbox_shape = (3,)
    self.bbox_data = np.zeros((self.batch_size, 3), dtype=np.float32)
    # face landmark
    self.landmark_shape = (10,)
    self.landmark_data = np.zeros((self.batch_size, 10), dtype=np.float32)
    # face mask
    self.mask_shape = (2,)
    self.mask_data = np.zeros((self.batch_size, 2), dtype=np.float32)

  def gen(self):
    """generate one batch, data layout [face, landmark, nonface]
    :return data: data in a batch
    :return bbox_data: bbox in a batch
    :return landmark_data: landmark in a batch
    :return mask_data: mask in a batch
    """
    # face
    start, end = 0, self.face_batch_size
    face_data, bbox_data = self.next_face_data().next()  # next() is needed for iter generator
    self.data[start:end] = face_data
    self.label_data[start:end] = 1
    self.bbox_data[start:end] = bbox_data
    self.landmark_data[start:end] = 0
    self.mask_data[start:end, 0] = 0
    self.mask_data[start:end, 1] = 1
    # landmark
    start, end = self.face_batch_size, self.face_batch_size+self.landmark_batch_size
    face_data, landmark_data = self.next_landmark_data().next()
    self.data[start:end] = face_data
    self.label_data[start:end] = 1
    self.bbox_data[start:end] = 0
    self.landmark_data[start:end] = landmark_data
    self.mask_data[start:end, 0] = 1
    self.mask_data[start:end, 1] = 0
    # nonface
    start, end = self.face_batch_size+self.landmark_batch_size, self.batch_size
    nonface_data = self.next_nonface_data().next()
    self.data[start:end] = nonface_data
    self.label_data[start:end] = 0
    self.bbox_data[start:end] = 0
    self.landmark_data[start:end] = 0
    self.mask_data[start:end] = 1
    # preprocess face_data
    self.data = preprocess_face_data(self.data)
    # copy it
    return (self.data.copy(), self.label_data.copy(),
            self.bbox_data.copy(), self.landmark_data.copy(),
            self.mask_data[:, 0].copy(),
            self.mask_data[:, 1].copy())

  def next_face_data(self):
    """generate face data in a batch
    """
    while True:
      self.face_generator.reset()
      for batch in self.face_generator:
        yield batch  # face_data, bbox_data

  def next_landmark_data(self):
    """generate landmark data in a batch
    """
    while True:
      self.landmark_generator.reset()
      for batch in self.landmark_generator:
        yield batch  # face_data, landmark_data

  def next_nonface_data(self):
    """generate nonface data in a batch
    """
    while True:
      self.nonface_generator.reset()
      for batch in self.nonface_generator:
        yield batch  # nonface_data

  def run(self):
    """run
    """
    while True:
      batch = self.gen()
      self.queue.put(batch)
