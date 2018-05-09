# pylint: disable=bad-indentation, no-member, invalid-name, line-too-long
import multiprocessing
import cv2
import lmdb
import numpy as np
from jfda.config import cfg


class MiniBatcher(multiprocessing.Process):
  '''generate minibatch
  given a queue, put (negatives, positives, part faces, landmark faces) = (n1, n2, n3, n4)
  '''

  def __init__(self, db_names, ns, net_type):
    '''order: negatives, positives, part faces, landmark faces
    '''
    super(MiniBatcher, self).__init__()
    self.ns = ns
    self.n = reduce(lambda x, acc: acc + x, ns, 0)
    self._start = [0 for _ in range(4)]
    self.net_type = net_type
    self.db_names = db_names
    self.db = [lmdb.open(db_name) for db_name in db_names]
    self.tnx = [db.begin() for db in self.db]
    self.db_size = [int(tnx.get('size')) for tnx in self.tnx]

  def __del__(self):
    for tnx in self.tnx:
      tnx.abort()
    for db in self.db:
      db.close()

  def set_queue(self, queue):
    self.queue = queue

  def get_size(self):
    return self.db_size

  def _make_transform(self, data, bbox=None, landmark=None):
    # gray scale
    if np.random.rand() < cfg.GRAY_PROB:
      gray = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
      data[:, :, 0] = gray
      data[:, :, 1] = gray
      data[:, :, 2] = gray
    # flip
    if np.random.rand() < cfg.FLIP_PROB:
      data = data[:, ::-1, :]
      if bbox is not None:
        # [dx1 dy1 dx2 dy2] --> [-dx2 dy1 -dx1 dy2]
        bbox[0], bbox[2] = -bbox[2], -bbox[0]
      if landmark is not None:
        landmark1 = landmark.reshape((-1, 2))
        # x --> 1 - x
        landmark1[:, 0] = 1 - landmark1[:, 0]
        landmark1[0], landmark1[1] = landmark1[1].copy(), landmark1[0].copy()
        landmark1[3], landmark1[4] = landmark1[4].copy(), landmark1[3].copy()
        landmark = landmark1.reshape(-1)
    data = data.transpose((2, 0, 1))
    return data, bbox, landmark

  def run(self):
    intpu_size = cfg.NET_INPUT_SIZE[self.net_type]
    data_shape = (intpu_size, intpu_size, 3)
    bbox_shape = (4,)
    landmark_shape = (10,)
    n = self.n
    while True:
      data = np.zeros((n, 3, intpu_size, intpu_size), dtype=np.float32)
      bbox_target = np.zeros((n, 4), dtype=np.float32)
      landmark_target = np.zeros((n, 10), dtype=np.float32)
      label = np.zeros(n, dtype=np.float32)

      start = self._start
      end = [start[i] + self.ns[i] for i in range(4)]
      for i in range(4):
        if end[i] > self.db_size[i]:
          end[i] -= self.db_size[i]
          start[i] = end[i]
          end[i] = start[i] + self.ns[i]

      idx = 0
      # negatives
      for i in xrange(start[0], end[0]):
        data_key = '%08d_data'%i
        _data = np.fromstring(self.tnx[0].get(data_key), dtype=np.uint8).reshape(data_shape)
        data[idx], _1, _2 = self._make_transform(_data)
        idx += 1
      # positives
      for i in xrange(start[1], end[1]):
        data_key = '%08d_data'%i
        bbox_key = '%08d_bbox'%i
        _data = np.fromstring(self.tnx[1].get(data_key), dtype=np.uint8).reshape(data_shape)
        _bbox_target = np.fromstring(self.tnx[1].get(bbox_key), dtype=np.float32).reshape(bbox_shape)
        data[idx], bbox_target[idx], _ = self._make_transform(_data, _bbox_target)
        idx += 1
      # part faces
      for i in xrange(start[2], end[2]):
        data_key = '%08d_data'%i
        bbox_key = '%08d_bbox'%i
        _data = np.fromstring(self.tnx[2].get(data_key), dtype=np.uint8).reshape(data_shape)
        _bbox_target = np.fromstring(self.tnx[2].get(bbox_key), dtype=np.float32).reshape(bbox_shape)
        data[idx], bbox_target[idx], _ = self._make_transform(_data, _bbox_target)
        idx += 1
      # landmark faces
      for i in range(start[3], end[3]):
        data_key = '%08d_data'%i
        landmark_key = '%08d_landmark'%i
        _data = np.fromstring(self.tnx[3].get(data_key), dtype=np.uint8).reshape(data_shape)
        _landmark_target = np.fromstring(self.tnx[3].get(landmark_key), dtype=np.float32).reshape(landmark_shape)
        data[idx], _, landmark_target[idx] = self._make_transform(_data, None, _landmark_target)
        idx += 1
      # label
      label[:self.ns[0]] = 0
      label[self.ns[0]: self.ns[0]+self.ns[1]] = 1
      label[self.ns[0]+self.ns[1]: self.ns[0]+self.ns[1]+self.ns[2]] = 2
      label[self.ns[0]+self.ns[1]+self.ns[2]:] = 3

      self._start = end
      data = (data - 128) / 128 # simple normalization
      minibatch = {'data': data,
                   'bbox_target': bbox_target,
                   'landmark_target': landmark_target,
                   'label': label}
      self.queue.put(minibatch)
