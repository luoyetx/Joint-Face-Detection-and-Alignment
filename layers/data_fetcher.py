#!/usr/bin/env python2.7
# coding = utf-8
#
# generate batch for the network.
#
# There're three type of data
#  1. face with bbox offset
#  2. face with landmark
#  3. non-face
#
# We generate this data in a batch with face_batch_size, landmark_batch_size and nonface_batch_size.
#  1. data, this is face data resized to 3 x ? x ?, where ? is 12 or 24 or 48
#  2. bbox, bbox offset, not all data type has this field
#  3. landmark, 5 landmark points, not all data type has this field
#  4. mask, this data field indicate the data point has bbox or landmark.
#     1. mask[0] for bbox offset and mask[1] for landmark
#     2. face with bbox offset, mask[0] = 0, mask[1] = 1, landmark data need to copy from network output
#     3. face with landmark offset, mask[0] = 1, mask[1] = 0, bbox offset data need to copy from network output
#     4. nonface, mask[0] = mask[1] = 1, both need to copy from network output
#  5. multiprocessing is used for speed up. Every data type has a process to load data.
#     Pipeline:
#       MiniBatch_Queue
#             ^                                      |<<<-------- FaceBatch_Queue --------<<< FaceGenerator
#             |                                      |
#             |<<<-------- BatchGenerator --------<<<|<<<-------- LandmarkBatch_Queue ----<<< LandmarkGenerator
#                                                    |
#                                                    |<<<-------- NonFaceBatch_Queue -----<<< NonFaceGenerator
#     Performance:
#       1. Random access the lmdb data can cause speed problem when no FaceBatch_Queue, etc.
#       2. Sequential access the lmdb data give the best performance.
#       3. The pipeline above gives almost the same performace with Random access or Seqential access the lmdb data.
#

import multiprocessing
import cv2
import lmdb
import numpy as np
from data.utils import get_face_size, load_wider, calc_IoU


NONFACE_OVERLAP_THRESHOLD = 0.3


def preprocess_face_data(face_data):
  """preprocess face data before feeding to network, original data lie in [0, 255]
  """
  res = (face_data - 128) / 128
  return res


class BaseGenerator(multiprocessing.Process):
  """Base Generator
  """

  def __init__(self, qout, db_name, net_type, batch_size, shuffle):
    super(BaseGenerator, self).__init__()
    self.qout = qout
    self.db_name = db_name
    self.net_type = net_type
    self.shuffle = shuffle
    self.batch_size = batch_size
    # load data
    self.db = lmdb.open(db_name)
    self.txn = self.db.begin()
    self.data_size = self.get_size()
    self.epoch_size = self.data_size / self.batch_size
    self.setup()
    # reset status
    self.start_idx = 0
    self.end_idx = 0
    self.shuffle_idx = np.arange(self.data_size)
    self.reset()

  def __del__(self):
    self.txn.abort()
    self.db.close()

  def setup(self):
    """subclass need to impl
    """
    pass

  def reset(self):
    if self.shuffle:
      self.start_idx = 0
      self.end_idx = self.start_idx + self.batch_size
      self.shuffle_idx = np.random.permutation(self.data_size)
    else:
      self.start_idx = self.end_idx
      self.end_idx = self.start_idx + self.batch_size
      self.shuffle_idx = np.arange(self.data_size)

  def get_mini_batch(self, idx):
    """subclass need to impl, generate mini-batch, raise StopIteration if end of datasize
    idx: data index
    return: (data-1, data-2, ...)
    """
    pass

  def get_size(self):
    """subclass need to impl, get data size
    txn: lmdb context
    return: full data size
    """
    size = int(self.txn.get('size'))
    return size

  def next(self):
    """
    return: mini-batch
    """
    self.end_idx = self.start_idx + self.batch_size
    if self.end_idx > self.data_size:
      self.end_idx -= self.data_size
      self.reset()  # reset start_idx and end_idx
    idx = self.shuffle_idx[self.start_idx:self.end_idx]
    self.start_idx = self.end_idx  # update
    mini_batch = self.get_mini_batch(idx)
    return mini_batch

  def run(self):
    """start process
    """
    while True:
      mini_batch = self.next()
      self.qout.put(mini_batch)


class FaceDataGenerator(BaseGenerator):
  """Face Data Generator
  given lmdb file path, generate a batch with size = batch_size each time
  """

  def setup(self):
    face_size = get_face_size(self.net_type)
    bbox_size = 3
    self.face_shape = (3, face_size, face_size)
    self.bbox_shape = (bbox_size,)
    self.face_data = np.zeros((self.batch_size, 3, face_size, face_size), dtype=np.float32)
    self.bbox_data = np.zeros((self.batch_size, bbox_size), dtype=np.float32)

  def get_mini_batch(self, idx):
    for i, key in enumerate(idx):
      face_key = '%08d_data'%key
      bbox_key = '%08d_offset'%key
      self.face_data[i] = np.fromstring(self.txn.get(face_key), dtype=np.uint8).reshape(self.face_shape)
      self.bbox_data[i] = np.fromstring(self.txn.get(bbox_key), dtype=np.float32).reshape(self.bbox_shape)
    return (self.face_data.copy(), self.bbox_data.copy())  # must copy due to Queue.put()


class LandmarkDataGenerator(BaseGenerator):
  """Landmark Data Generator
  """

  def setup(self):
    face_size = get_face_size(self.net_type)
    landmark_size = 10
    self.face_shape = (3, face_size, face_size)
    self.landmark_shape = (landmark_size,)
    self.face_data = np.zeros((self.batch_size, 3, face_size, face_size), dtype=np.float32)
    self.landmark_data = np.zeros((self.batch_size, landmark_size), dtype=np.float32)

  def get_mini_batch(self, idx):
    for i, key in enumerate(idx):
      face_key = '%08d_data'%key
      landmark_key = '%08d_landmark'%key
      self.face_data[i] = np.fromstring(self.txn.get(face_key), dtype=np.uint8).reshape(self.face_shape)
      self.landmark_data[i] = np.fromstring(self.txn.get(landmark_key), dtype=np.float32).reshape(self.landmark_shape)
    return (self.face_data.copy(), self.landmark_data.copy())  # must copy due to Queue.put()


class NonFaceDataGenerator(BaseGenerator):
  """Nonface Data Generator
  """

  def setup(self):
    nonface_size = get_face_size(self.net_type)
    self.nonface_shape = (3, nonface_size, nonface_size)
    self.nonface_data = np.zeros((self.batch_size, 3, nonface_size, nonface_size), dtype=np.float32)

  def get_mini_batch(self, idx):
    for i, key in enumerate(idx):
      nonface_key = '%08d_data'%key
      self.nonface_data[i] = np.fromstring(self.txn.get(nonface_key), dtype=np.uint8).reshape(self.nonface_shape)
    return (self.nonface_data.copy())  # must copy due to Queue.put()


class BgNonFaceDataGenerator(multiprocessing.Process):
  """Nonface Data Generator from image data
  """

  def __init__(self, qout, bgs, net_type='p', batch_size=1024, shuffle=False):
    """init
    :param bgs: bg list with bbox, [(img_path, bboxes), (img_path, bboxes)....]
    """
    super(BgNonFaceDataGenerator, self).__init__()
    self.qout = qout
    self.bgs = bgs
    self.data_size = len(bgs)
    self.epoch_size = self.data_size
    self.batch_size = batch_size
    self.current_bg_idx = 0
    self.shuffle = shuffle
    self.shuffle_idx = []
    self.reset()
    size = get_face_size(net_type)
    self.nonface_shape = (3, size, size)
    self.nonface_data = np.zeros((batch_size, 3, size, size), dtype=np.float32)

  def reset(self):
    self.current_bg_idx = 0
    if self.shuffle:
      self.shuffle_idx = np.random.permutation(self.data_size)
    else:
      self.shuffle_idx = np.arange(self.data_size)

  def next(self):
    if self.current_bg_idx >= self.data_size:
      self.reset()
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
    return (self.nonface_data.copy())

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
        if calc_IoU(nonface_bbox, face_bbox) > NONFACE_OVERLAP_THRESHOLD:
          use_it = False
          break
      if use_it:
        nonface_bboxes.append(nonface_bbox)
    return nonface_bboxes

  def run(self):
    while True:
      mini_batch = self.next()
      self.qout.put(mini_batch)


class BatchGenerator(multiprocessing.Process):
  """generate batches from given dataset, put the batch in the queue for MTDataIter
  """

  def __init__(self, queue, net_type='p',
               is_train=True, shuffle=False,
               face_db_name='data/pnet_face_train',
               landmark_db_name='data/pnet_landmark_train',
               nonface_db_name='data/pnet_nonface_train',
               face_batch_size=256,
               landmark_batch_size=256,
               nonface_batch_size=1024):
    super(BatchGenerator, self).__init__()
    self.q_to = queue
    self.q_from = [multiprocessing.Queue(32) for i in range(3)]
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
    # create generator
    self.generator = []
    self.generator.append(FaceDataGenerator(self.q_from[0],
                                            db_name=face_db_name,
                                            net_type=net_type,
                                            batch_size=face_batch_size,
                                            shuffle=shuffle))
    self.generator.append(LandmarkDataGenerator(self.q_from[1],
                                                db_name=landmark_db_name,
                                                net_type=net_type,
                                                batch_size=landmark_batch_size,
                                                shuffle=shuffle))
    if nonface_db_name == 'None':
      train_data, val_data = load_wider()
      if is_train is True:
        bgs = train_data
      else:
        bgs = val_data
      self.generator.append(BgNonFaceDataGenerator(self.q_from[2],
                                                   bgs=bgs,
                                                   net_type=net_type,
                                                   batch_size=nonface_batch_size,
                                                   shuffle=shuffle))
    else:
      self.generator.append(NonFaceDataGenerator(self.q_from[2],
                                                 db_name=nonface_db_name,
                                                 net_type=net_type,
                                                 batch_size=nonface_batch_size,
                                                 shuffle=shuffle))
    # launch processes
    for g in self.generator:
      g.start()

    def cleanup():
      for g in self.generator:
        g.terminate()
      for g in self.generator:
        g.join()
    import atexit
    atexit.register(cleanup)

  def gen(self):
    """generate one batch, data layout [face, landmark, nonface]
    :return data: data in a batch
    :return bbox_data: bbox in a batch
    :return landmark_data: landmark in a batch
    :return mask_data: mask in a batch
    """
    # face
    start, end = 0, self.face_batch_size
    face_data, bbox_data = self.next_face_data()
    self.data[start:end] = face_data
    self.label_data[start:end] = 1
    self.bbox_data[start:end] = bbox_data
    self.landmark_data[start:end] = 0
    self.mask_data[start:end, 0] = 0
    self.mask_data[start:end, 1] = 1
    # landmark
    start, end = self.face_batch_size, self.face_batch_size+self.landmark_batch_size
    face_data, landmark_data = self.next_landmark_data()
    self.data[start:end] = face_data
    self.label_data[start:end] = 1
    self.bbox_data[start:end] = 0
    self.landmark_data[start:end] = landmark_data
    self.mask_data[start:end, 0] = 1
    self.mask_data[start:end, 1] = 0
    # nonface
    start, end = self.face_batch_size+self.landmark_batch_size, self.batch_size
    nonface_data = self.next_nonface_data()
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
            self.mask_data[:, 1].copy())  # must copy due to Queue.put()

  def next_face_data(self):
    """generate face data in a batch
    """
    # face_data, bbox_data
    return self.q_from[0].get()

  def next_landmark_data(self):
    """generate landmark data in a batch
    """
    # face_data, landmark_data
    return self.q_from[1].get()

  def next_nonface_data(self):
    """generate nonface data in a batch
    """
    # nonface_data
    return self.q_from[2].get()

  def run(self):
    """run
    """
    while True:
      batch = self.gen()
      self.q_to.put(batch)
