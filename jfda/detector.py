#!/usr/bin/env python2.7
# coding = utf-8

import os
os.environ['GLOG_minloglevel'] = '2'

import cv2
import caffe
import numpy as np
from .utils import nms


FACE_PROB_TH = 0.8


class Detector(object):
  """Cascaded CNN Face Detector
  """
  def __init__(self, net, model):
    """init detector
    """
    self.pnet = caffe.Net(net, caffe.TEST, weights=model)

  def detect(self, img):
    """detect face
    """
    height, width = img.shape[:-1]
    scale = 1
    factor = 1.4
    win_min_size = 20
    face = []
    while min(height, width) >= win_min_size:
      data = img.transpose(2, 0, 1)
      data = (data.astype(np.float32) - 128) / 128
      self.pnet.blobs['data'].reshape(1, 3, height, width)
      self.pnet.blobs['data'].data[...] = data

      output = self.pnet.forward()
      prob = output['face_prob'][0][1]
      bbox_offset = output['face_bbox'][0]
      landmark = output['face_landmark'][0]
      rheight, rwidth = prob.shape
      counter = 0
      for i in range(rheight):
        for j in range(rwidth):
          if prob[i][j] > FACE_PROB_TH:
            x, y, w, h = 2*j*scale, 2*i*scale, 12*scale, 12*scale
            dx, dy, ds = bbox_offset[:, i, j]
            x_ = x + w * dx
            y_ = y + h * dy
            w_ = w * np.exp(ds)
            h_ = h * np.exp(ds)
            bbox = np.array([x_, y_, w_, h_, prob[i][j]])
            face.append(bbox)
            counter += 1
      height = int(height/factor)
      width = int(width/factor)
      scale *= factor
      img = cv2.resize(img, (width, height))
    face = np.asarray(face)
    keep = nms(face, 0.3)
    face = face[keep, :].copy()
    return face


if __name__ == '__main__':
  detector = Detector('proto/p.prototxt', 'result/p.caffemodel')
  img = cv2.imread('test.jpg', cv2.IMREAD_COLOR)
  face = detector.detect(img)
  for bbox in face:
    x, y, w, h, s = bbox
    cv2.rectangle(img, (int(x), int(y)), (int(x+w), int(y+h)), (0, 0, 255), 1)
    # for x, y in landmark:
    #   cv2.circle(img, (int(x), int(y)), 2, (0, 255, 0), -1)
  cv2.imwrite('res.jpg', img)
