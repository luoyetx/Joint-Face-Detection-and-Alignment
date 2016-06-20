# coding = utf-8

import os
import logging
import numpy as np
from config import config


def get_dirmapper(dirpath):
  '''return dir mapper for wider face
  '''
  mapper = {}
  for d in os.listdir(dirpath):
    dir_id = d.split('--')[0]
    mapper[dir_id] = os.path.join(dirpath, d)
  return mapper


def load_wider():
  '''load wider face dataset
  '''
  train_mapper = get_dirmapper(os.path.join(config['data_wider'], 'WIDER_train/images'))
  val_mapper = get_dirmapper(os.path.join(config['data_wider'], 'WIDER_val/images'))

  def gen(text, mapper):
    fin = open(text, 'r')

    img_paths = []
    bboxes = []
    while True:
      line = fin.readline()
      if not line: break  # eof
      name = line.strip()
      dir_id = name.split('_')[0]
      fpath = os.path.join(mapper[dir_id], name + '.jpg')
      face_n = int(fin.readline().strip())

      internal_bboxes = []
      for i in range(face_n):
        line = fin.readline().strip()
        components = line.split(' ')
        x, y, w, h = [float(_) for _ in components]

        # refine bbox
        size = (w + h) / 2
        x_center = x + w / 2
        y_center = y + h / 2
        x = x_center - size / 2
        y = y_center - size / 2
        w = h = size
        x, y, w, h = int(x), int(y), int(w), int(h)
        bbox = [x, y, w, h]
        internal_bboxes.append(bbox)

      img_paths.append(fpath)
      bboxes.append(internal_bboxes)
    fin.close()
    return (img_paths, bboxes)

  train_data = gen('wider_face_train.txt', train_mapper)
  val_data = gen('wider_face_val.txt', val_mapper)
  return (train_data, val_data)


def load_celeba():
  '''load celeba dataset and crop the face bbox
  notice: the face bbox may out of the image range
  '''
  text = os.path.join(config['data_celeba'], 'list_landmarks_celeba.txt')
  fin = open(text, 'r')
  n = int(fin.readline().strip())
  fin.readline()  # drop

  img_paths = []
  landmarks = []
  bboxes = []
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

    img_paths.append(img_path)
    landmarks.append(landmark)
    bboxes.append(bbox)

  fin.close()
  return (img_paths, landmarks, bboxes)


def calc_IoU(bbox1, bbox2):
  '''calculate IoU of bbox1 and bbox2
  '''
  area1 = bbox1[2] * bbox1[3]
  area2 = bbox2[2] * bbox2[3]
  x1, y1 = max(bbox1[0], bbox2[0]), max(bbox1[1], bbox2[1])
  x2, y2 = min(bbox1[0]+bbox1[2], bbox2[0]+bbox2[2]), min(bbox1[1]+bbox1[3], bbox2[1]+bbox2[3])
  w, h = max(0, x2-x1), max(0, y2-y1)
  area_overlap = w * h
  return float(area_overlap) / (area1 + area2 - area_overlap)


def check_bbox(bbox, region):
  '''check the bbox if out of the region
  '''
  x, y, w, h = bbox
  height, width = region
  if x < 0 or y < 0 or x+w-1 >= width or y+h-1 >= height:
    return False
  else:
    return True


def get_logger(name=None):
  '''return a logger
  '''
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
