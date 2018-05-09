import json
import caffe
from jfda.config import cfg


class FaceDataLayer(caffe.Layer):
  '''Custom Data Layer
  LayerOutput
    top[0]: image data
    top[1]: bbox target
    top[2]: landmark target
    top[3]: face data type / label, 0 for negatives, 1 for positives
                                    2 for part faces, 3 for landmark faces

  Howto
    layer {
      name: "data"
      type: "Python"
      top: "data"
      top: "bbox_target"
      top: "landmark_target"
      top: "label"
      python_param {
        module: "layers.data_layer"
        layer: "FaceDataLayer"
      }
    }
  '''

  def set_batch_num(self, n1, n2, n3, n4):
    '''set data type number
    n1 for negatives, n2 for positives, n3 for part faces, n4 for landmark faces
    net_input_size for network input size (width, height)
    '''
    self.n1 = n1
    self.n2 = n2
    self.n3 = n3
    self.n4 = n4
    self.n = n1 + n2 + n3 + n4
    self.net_input_size = cfg.NET_INPUT_SIZE[cfg.NET_TYPE]

  def set_data_queue(self, queue):
    '''the queue should put a minibatch with size of (negatives, positives, part faces, landmark faces) =
    (n1, n2, n3, n4) in a dict
    '''
    self.data_queue = queue

  def setup(self, bottom, top):
    self.n1 = 1
    self.n2 = 1
    self.n3 = 1
    self.n4 = 1
    self.n = 4
    self.net_input_size = cfg.NET_INPUT_SIZE[cfg.NET_TYPE]
    self.reshape(bottom, top)

  def reshape(self, bottom, top):
    top[0].reshape(self.n, 3, self.net_input_size, self.net_input_size)
    top[1].reshape(self.n, 4)
    top[2].reshape(self.n, 10)
    top[3].reshape(self.n)

  def forward(self, bottom, top):
    minibatch = self._get_minibacth()
    # face data
    top[0].data[...] = minibatch['data']
    top[1].data[...] = minibatch['bbox_target']
    top[2].data[...] = minibatch['landmark_target']
    top[3].data[...] = minibatch['label']

  def backward(self, bottom, top):
    pass

  def _get_minibacth(self):
    minibatch = self.data_queue.get()
    return minibatch
