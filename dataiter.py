#!/usr/bin/env python2.7
# coding = utf-8

import multiprocessing
import cv2
import lmdb
import numpy as np
import mxnet as mx
from data.utils import load_wider
from data.utils import calc_IoU


NONFACE_OVERLAP_THRESHOLD = 0.3
BATCH_QUEUE_LEN = 10
batch_queue = multiprocessing.Queue(BATCH_QUEUE_LEN)  # used for batches


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
        if calc_IoU(nonface_bbox, face_bbox) > NONFACE_OVERLAP_THRESHOLD:
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
    self.label_shape = 1
    self.label_data = np.zeros((self.batch_size,), dtype=np.float32)
    self.bbox_shape = 4
    self.bbox_data = np.zeros((self.batch_size, 4), dtype=np.float32)
    self.landmark_shape = 10
    self.landmark_data = np.zeros((self.batch_size, 10), dtype=np.float32)
    self.mask_shape = 2
    self.mask_data = np.zeros((self.batch_size, 2), dtype=np.float32)

  def finalize(self):
    del self.face_generator
    del self.landmark_generator
    del self.nonface_generator

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
    self.label_data[start:end] = 1
    self.bbox_data[start:end] = bbox_data
    self.landmark_data[start:end].fill(0)  # must fill 0
    self.mask_data[start:end] = mask_data
    # landmark
    start, end = self.face_batch_size, self.face_batch_size+self.landmark_batch_size
    face_data, landmark_data, mask_data = self.next_landmark_data().next()
    self.data[start:end] = face_data
    self.label_data[start:end] = 1
    self.bbox_data[start:end].fill(0)
    self.landmark_data[start:end] = landmark_data
    self.mask_data[start:end] = mask_data
    # nonface
    start, end = self.face_batch_size+self.landmark_batch_size, self.batch_size
    nonface_data, mask_data = self.next_nonface_data().next()
    self.data[start:end] = nonface_data
    self.label_data[start:end] = 0
    self.bbox_data[start:end].fill(0)
    self.landmark_data[start:end].fill(0)
    self.mask_data[start:end] = mask_data
    # copy it
    return (self.data.copy(), self.label_data.copy(),
            self.bbox_data.copy(), self.landmark_data.copy(),
            self.mask_data[:, 0].copy(), self.mask_data[:, 1].copy())

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


def bgt_wrapper(bq, kwargs):
  """wrap BatchGenerator for Process target,
  """
  bgt = BatchGenerator(**kwargs)
  while True:
    batch = bgt.run()
    bq.put(batch)
  # will never come here, need a way to safely shut down
  # but the resource will still release through method __del__
  bgt.finalize()


class MTDataIter(mx.io.DataIter):
  """Multi-task DataIter
  """

  def __init__(self, net_type='p', is_train=True,
               shuffle=False, epoch_size=1000,
               batch_sizes=[256, 256, 512]):
    """init the data iter
    :param net_type: 'p' or 'r' or 'o' for 3 nets
    :param batch_sizes: 3 batch_sizes for face, landmark, non-face
    """
    super(MTDataIter, self).__init__()
    assert net_type in ['p', 'r', 'o']
    self.net_type = net_type
    self.face_batch_size = batch_sizes[0]
    self.landmark_batch_size = batch_sizes[1]
    self.nonface_batch_size = batch_sizes[2]
    self.batch_size = reduce(lambda acc, x: acc+x, batch_sizes, 0)
    data_type = 'train' if is_train else 'val'
    face_db_name = 'data/%snet_face_%s'%(net_type, data_type)
    landmark_db_name = 'data/%snet_landmark_%s'%(net_type, data_type)
    train_list, val_list = load_wider()
    nonface_bgs = train_list if is_train else val_list
    kwargs = {
      'net_type': net_type,
      'shuffle': shuffle,
      'face_db_name': face_db_name,
      'landmark_db_name': landmark_db_name,
      'nonface_bgs': nonface_bgs,
      'face_batch_size': batch_sizes[0],
      'landmark_batch_size': batch_sizes[1],
      'nonface_batch_size': batch_sizes[2],
    }

    self.generator = multiprocessing.Process(target=bgt_wrapper,
                                             args=(batch_queue, kwargs))
    self.generator.start()
    self.current_batch_idx = 0
    self.epoch_size = epoch_size
    self.reset()

    if net_type == 'p':
      self.data_shape =  (self.batch_size, 3, 12, 12)
    elif net_type == 'r':
      self.data_shape = (self.batch_size, 3, 24, 24)
    else:
      self.data_shape = (self.batch_size, 3, 48, 48)
    self.face_shape = (self.batch_size, 1)
    self.bbox_shape = (self.batch_size, 4)
    self.landmark_shape = (self.batch_size, 10)
    self.bbox_mask_shape = (self.batch_size, 1)
    self.landmark_mask_shape = (self.batch_size, 1)

  def finalize(self):
    # forcely shut down
    self.generator.terminate()

  def get_one_batch(self):
    """for debug
    """
    batch = batch_queue.get()
    return batch

  def reset(self):
    self.current_batch_idx = 0

  def iter_next(self):
    if self.current_batch_idx >= self.epoch_size:
      return False
    self.batch = batch_queue.get()
    self.data = [mx.nd.array(data) for data in self.batch[:1]]
    self.label = [mx.nd.array(data) for data in self.batch[1:]]
    self.current_batch_idx += 1
    return True

  def getdata(self):
    return self.data

  def getlabel(self):
    return self.label

  def getindex(self):
    return self.current_batch_idx

  def getpad(self):
    return 0

  @property
  def provide_data(self):
    return [('data', self.data_shape)]

  @property
  def provide_label(self):
    return [('label', self.label_shape),
            ('bbox', self.bbox_shape),
            ('landmark', self.landmark_shape),
            ('bbox_mask', self.bbox_mask_shape),
            ('landmark_mask', self.landmark_mask_shape)]


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
    print 'batch', idx
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
  bgt.finalize()
  # test MTDataIter
  data_iter = MTDataIter(epoch_size=30)
  batch = data_iter.get_one_batch()
  print 'get one'
  assert batch[0].shape == (1024, 3, 12, 12)  # data
  assert batch[1].shape == (1024,)  # face cls
  assert batch[2].shape == (1024, 4)  # bbox rg
  assert batch[3].shape == (1024, 10)  # landmark rg
  assert batch[4].shape == (1024,)  # bbox mask
  assert batch[5].shape == (1024,)  # landmark mask
  batch = data_iter.get_one_batch()
  print 'get one'
  # test MTDataIter
  data_iter.reset()
  for idx, batch in enumerate(data_iter):
    print 'batch', idx
  data_iter.finalize()
