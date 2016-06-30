#!/usr/bin/env python2.7
# coding = utf-8

import multiprocessing
import cv2
import lmdb
import numpy as np
import mxnet as mx
from data.utils import load_wider
from data.fetcher import BatchGenerator


BATCH_QUEUE_LEN = 10
batch_queue = multiprocessing.Queue(BATCH_QUEUE_LEN)  # used for batches


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

    self.generator = BatchGenerator(batch_queue, **kwargs)
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
    self.face_shape = (self.batch_size,)
    self.bbox_shape = (self.batch_size, 4)
    self.landmark_shape = (self.batch_size, 10)
    self.bbox_mask_shape = (self.batch_size, 1)
    self.landmark_mask_shape = (self.batch_size, 1)

    def cleanup():
      self.generator.terminate()
      self.generator.join()
    import atexit
    atexit.register(cleanup)


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
    return [('face', self.face_shape),
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
  assert batch[4].shape == (1024, 1)  # bbox mask
  assert batch[5].shape == (1024, 1)  # landmark mask
  batch = data_iter.get_one_batch()
  print 'get one'
  # test MTDataIter
  data_iter.reset()
  for idx, batch in enumerate(data_iter):
    print 'batch', idx
  data_iter.finalize()
