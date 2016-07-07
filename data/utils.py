#!/usr/bin/env python2.7
# coding = utf-8

import os
import logging
import cv2
import numpy as np


# dataDir contains `WIDER`, `CelebA`, `aflw`
data_dir = os.path.dirname(os.path.abspath(__file__))
# dataWIDER contains `WIDER_train`, `WIDER_val`, `WIDER_test`, `wider_face_split`
data_wider = os.path.join(data_dir, 'WIDER')
# dataCelebA contains `img_celeba`, `list_landmarks_celeba.txt`
data_celeba = os.path.join(data_dir, 'CelebA')

config = {
  'data_dir': data_dir,
  'data_wider': data_wider,
  'data_celeba': data_celeba
}


def get_face_size(net_type):
  """return face size according to net_type
  """
  if net_type == 'p':
    return 12
  elif net_type == 'r':
    return 24
  else:
    assert net_type == 'o'
    return 48


def get_dirmapper(dirpath):
  """return dir mapper for wider face
  """
  mapper = {}
  for d in os.listdir(dirpath):
    dir_id = d.split('--')[0]
    mapper[dir_id] = os.path.join(dirpath, d)
  return mapper


def load_wider():
  """load wider face dataset
  """
  train_mapper = get_dirmapper(os.path.join(config['data_wider'], 'WIDER_train', 'images'))
  val_mapper = get_dirmapper(os.path.join(config['data_wider'], 'WIDER_val', 'images'))

  def gen(text, mapper):
    fin = open(text, 'r')

    result = []
    while True:
      line = fin.readline()
      if not line: break  # eof
      name = line.strip()
      dir_id = name.split('_')[0]
      img_path = os.path.join(mapper[dir_id], name + '.jpg')
      face_n = int(fin.readline().strip())

      bboxes = []
      for i in range(face_n):
        line = fin.readline().strip()
        components = line.split(' ')
        x, y, w, h = [float(_) for _ in components]
        x_center = x + w / 2
        y_center = y + h / 2
        size = (w + h) / 2
        x = x_center - size / 2
        y = y_center - size / 2

        # only large enough
        if size > 20:
          bbox = [int(x), int(y), int(size), int(size)]
          bboxes.append(bbox)

      # # for debug
      # img = cv2.imread(img_path)
      # for bbox in bboxes:
      #   x, y, w, h = bbox
      #   cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 1)
      # cv2.imshow('img', img)
      # cv2.waitKey(0)

      result.append([img_path, bboxes])
    fin.close()
    return result

  txt_dir = os.path.join(config['data_wider'], 'wider_face_split')
  train_data = gen(os.path.join(txt_dir, 'wider_face_train.txt'), train_mapper)
  val_data = gen(os.path.join(txt_dir, 'wider_face_val.txt'), val_mapper)
  return (train_data, val_data)


def load_celeba():
  """load celeba dataset and crop the face bbox
  notice: the face bbox may out of the image range
  """
  text = os.path.join(config['data_celeba'], 'list_landmarks_celeba.txt')
  fin = open(text, 'r')
  n = int(fin.readline().strip())
  fin.readline()  # drop

  result = []
  for i in range(n):
    line = fin.readline().strip()
    components = line.split()
    img_path = os.path.join(config['data_celeba'], 'img_celeba', components[0])
    landmark = np.asarray([int(_) for _ in components[1:]], dtype=np.float32)
    landmark = landmark.reshape(len(landmark)/2, 2)

    # crop face bbox
    x_max, y_max = landmark.max(0)
    x_min, y_min = landmark.min(0)
    w, h = x_max - x_min, y_max - y_min
    w = h = max(w, h)
    ratio = 0.5
    x_new = x_min - w*ratio
    y_new = y_min - h*ratio
    w_new = w*(1 + 2*ratio)
    h_new = h*(1 + 2*ratio)
    bbox = [x_new, y_new, w_new, h_new]
    bbox = [int(_) for _ in bbox]

    # # for debug
    # img = cv2.imread(img_path)
    # x, y, w, h = bbox
    # cv2.rectangle(img, (x, y), (x+w, y+h), (0,0,255), 1)
    # for j in range(5):
    #   cv2.circle(img, (int(landmark[j, 0]), int(landmark[j, 1])), 2, (0,255,0), -1)
    # cv2.imshow('img', img)
    # cv2.waitKey(0)

    # normalize landmark
    landmark[:, 0] = (landmark[:, 0] - bbox[0]) / bbox[2]
    landmark[:, 1] = (landmark[:, 1] - bbox[1]) / bbox[3]

    landmark = landmark.reshape(2*len(landmark))
    result.append([img_path, bbox, landmark])

  fin.close()
  return result


def calc_IoU(bbox1, bbox2):
  """calculate IoU of bbox1 and bbox2
  """
  area1 = bbox1[2] * bbox1[3]
  area2 = bbox2[2] * bbox2[3]
  x1, y1 = max(bbox1[0], bbox2[0]), max(bbox1[1], bbox2[1])
  x2, y2 = min(bbox1[0]+bbox1[2], bbox2[0]+bbox2[2]), min(bbox1[1]+bbox1[3], bbox2[1]+bbox2[3])
  w, h = max(0, x2-x1), max(0, y2-y1)
  area_overlap = w * h
  return float(area_overlap) / (area1 + area2 - area_overlap)


def calc_IoUs(bbox, bboxes):
  """calculate IoUs between bbox and bboxes
  """
  IoUs = np.ones(10)
  for idx, bbox_ in bboxes:
    IoUs[idx] = calc_IoU(bbox, bbox_)
  return IoUs


def check_bbox(bbox, region):
  """check the bbox if out of the region
  """
  x, y, w, h = bbox
  height, width = region
  if x < 0 or y < 0 or x+w-1 >= width or y+h-1 >= height:
    return False
  else:
    return True


def draw_landmark(img_, bbox, landmark):
  """for debug
  """
  img = np.copy(img_)
  x, y, w, h = bbox
  cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
  landmark_n = len(landmark) / 2
  for i in range(landmark_n):
    cv2.circle(img, (int(x+w*landmark[2*i]), int(y+h*landmark[2*i+1])), 2, (0, 255, 0), -1)
  return img


def show_image(img):
  """for debug
  """
  cv2.imshow('img', img)
  cv2.waitKey(0)


def get_logger(name=None):
  """return a logger
  """
  logger = logging.getLogger(name)
  logger.setLevel(logging.INFO)
  sh = logging.StreamHandler()
  sh.setLevel(logging.INFO)
  formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s')
  sh.setFormatter(formatter)
  logger.addHandler(sh)
  return logger


if __name__ == '__main__':
  pass
