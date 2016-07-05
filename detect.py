#!/usr/bin/env python2.7
# coding = utf-8

import cv2
import caffe
import numpy as np


FACE_PROB_TH = 0.8


def nms(dets, thresh):
  x1 = dets[:, 0]
  y1 = dets[:, 1]
  x2 = dets[:, 0] + dets[:, 2]
  y2 = dets[:, 1] + dets[:, 3]
  scores = dets[:, 4]

  areas = (x2 - x1 + 1) * (y2 - y1 + 1)
  order = scores.argsort()[::-1]

  keep = []
  while order.size > 0:
    i = order[0]
    keep.append(i)
    xx1 = np.maximum(x1[i], x1[order[1:]])
    yy1 = np.maximum(y1[i], y1[order[1:]])
    xx2 = np.minimum(x2[i], x2[order[1:]])
    yy2 = np.minimum(y2[i], y2[order[1:]])

    w = np.maximum(0.0, xx2 - xx1 + 1)
    h = np.maximum(0.0, yy2 - yy1 + 1)
    inter = w * h
    ovr = inter / (areas[i] + areas[order[1:]] - inter)

    inds = np.where(ovr <= thresh)[0]
    order = order[inds + 1]

  return keep


class Detector(object):
  """Cascaded CNN Face Detector
  """
  def __init__(self):
    """init detector
    """
    self.pnet = caffe.Net('proto/p.prototxt', 'result/p.caffemodel', caffe.TEST)

  def detect(self, img):
    """detect face
    """
    height, width = img.shape[:-1]
    scale_factor = 1
    factor = 1.4
    win_min_size = 20
    face = []
    while min(height, width) >= win_min_size:
      data = img.transpose(2, 0, 1)
      single_face = self.detect_single_scale(data)
      print scale_factor, len(single_face['bbox'])
      for bbox in single_face['bbox']:
        x, y, w, h, s = bbox
        x *= scale_factor
        y *= scale_factor
        w *= scale_factor
        h *= scale_factor
        bbox_new = [x, y, w, h, s]
        face.append(bbox_new)
      height /= factor
      width /= factor
      scale_factor *= factor
      img = cv2.resize(img, (int(height), int(width)))
    face = np.asarray(face)
    keep = nms(face, 0.3)
    face = face[keep, :]
    return face

  def detect_single_scale(self, data):
    """detect single scale face
    """
    # preprocess data
    data = (data.astype(np.float32) - 128) / 128
    height, width = data.shape[1:]
    self.pnet.blobs['data'].reshape(1, 3, height, width)
    self.pnet.blobs['data'].data[...] = data
    output = self.pnet.forward()
    prob = output['face_prob'][0][1]
    bbox_offset = output['face_bbox'][0]
    landmark = output['face_landmark'][0]
    height, width = prob.shape

    face = {'bbox': [], 'landmark': []}
    for i in range(height):
      for j in range(width):
        if prob[i][j] > FACE_PROB_TH:
          x, y, w, h = 2*j, 2*i, 12, 12
          dx, dy, dw, dh = bbox_offset[:, i, j]
          x_ = x + w * dx
          y_ = y + h * dy
          w_ = w * dw
          h_ = h * dh
          bbox = np.array([x, y, w, h, prob[i][j]])
          face['bbox'].append(bbox)
          face['landmark'].append(landmark[:, i, j])
    face['bbox'] = np.asarray(face['bbox'])
    face['landmark'] = np.asarray(face['landmark'])
    return face


if __name__ == '__main__':
  detector = Detector()
  img = cv2.imread('test.jpg', cv2.IMREAD_COLOR)
  # h, w = img.shape[:-1]
  # img = cv2.resize(img, (w/2, h/2))
  face = detector.detect(img)
  for bbox in face:
    x, y, w, h, s = bbox
    cv2.rectangle(img, (int(x), int(y)), (int(x+w), int(y+h)), (0, 0, 255), 1)
    # for x, y in landmark:
    #   cv2.circle(img, (int(x), int(y)), 2, (0, 255, 0), -1)
  cv2.imwrite('res.jpg', img)
