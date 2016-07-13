# coding = utf-8

import json
from multiprocessing import Queue
import caffe
from data.utils import get_face_size
from data.fetcher import BatchGenerator


class JfdaDataLayer(caffe.Layer):
  """Custom Data Layer

  How to
  ======

  write the layer like below in prototxt

  layer {
    name: "data"
    type: "Python"
    top: "data"
    top: "label"
    top: "bbox"
    top: "landmark"
    top: "bbox_mask"
    top: "landmark_mask"
    python_param {
      module: "layers.jfda_data_layer"
      layer: "JfdaDataLayer"
      param_str: '{'
                 '  "net_type": "p",'
                 '  "face_db_name": "data/pnet_face_train",'
                 '  "landmark_db_name": "data/pnet_landmark_train",'
                 '  "nonface_db_name": "data/pnet_nonface_train",'
                 '  "face_batch_size": 256,'
                 '  "landmark_batch_size": 256,'
                 '  "nonface_batch_size": 512'
                 '}'
    }
  }
  """

  def setup(self, bottom, top):
    """set up this data layer, load params and launch generator
    """
    param = json.loads(self.param_str)
    net_type = param['net_type']
    face_db_name = param['face_db_name']
    landmark_db_name = param['landmark_db_name']
    nonface_db_name = param['nonface_db_name']
    face_batch_size = param['face_batch_size']
    landmark_batch_size = param['landmark_batch_size']
    nonface_batch_size = param['nonface_batch_size']

    assert net_type == 'p' or \
           net_type == 'r' or \
           net_type == 'o'

    kwargs = {
      'net_type': net_type,
      'face_db_name': face_db_name,
      'landmark_db_name': landmark_db_name,
      'nonface_db_name': nonface_db_name,
      'face_batch_size': face_batch_size,
      'landmark_batch_size': landmark_batch_size,
      'nonface_batch_size': nonface_batch_size,
    }
    # launch generator
    self.queue = Queue(32)
    self.generator = BatchGenerator(self.queue, **kwargs)
    self.generator.start()

    def cleanup():
      self.generator.terminate()
      self.generator.join()
    import atexit
    atexit.register(cleanup)

    self.batch_size = face_batch_size + landmark_batch_size + nonface_batch_size
    self.face_size = get_face_size(net_type)
    top[0].reshape(self.batch_size, 3, self.face_size, self.face_size)
    top[1].reshape(self.batch_size)
    top[2].reshape(self.batch_size, 3)
    top[3].reshape(self.batch_size, 10)
    top[4].reshape(self.batch_size)
    top[5].reshape(self.batch_size)

  def reshape(self, bottom, top):
    """already reshape in function setup
    """
    pass

  def forward(self, bottom, top):
    """forward to feed the network
    """
    batch = self.get_mini_batch()
    # face data
    top[0].data[...] = batch[0]
    # label
    top[1].data[...] = batch[1]
    # bbox
    top[2].data[...] = batch[2]
    # landmark
    top[3].data[...] = batch[3]
    # bbox mask
    top[4].data[...] = batch[4]
    # landmark mask
    top[5].data[...] = batch[5]

  def backward(self, top, propagte_down, bottom):
    """data layer don't need backward
    """
    pass

  def get_mini_batch(self):
    batch = self.queue.get()
    return batch
