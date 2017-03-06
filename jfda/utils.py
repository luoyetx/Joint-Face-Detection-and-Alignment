# pylint: disable=bad-indentation, no-member, invalid-name, line-too-long
import os
import time
import logging
import cv2
import numpy as np
from jfda.config import cfg


def load_wider():
  """load wider face dataset
  data: [img_path, bboxes]+
  bboxes: [x1, y1, x2, y2]
  """

  def get_dirmapper(dirpath):
    """return dir mapper for wider face
    """
    mapper = {}
    for d in os.listdir(dirpath):
      dir_id = d.split('--')[0]
      mapper[dir_id] = os.path.join(dirpath, d)
    return mapper

  train_mapper = get_dirmapper(os.path.join(cfg.WIDER_DIR, 'WIDER_train', 'images'))
  val_mapper = get_dirmapper(os.path.join(cfg.WIDER_DIR, 'WIDER_val', 'images'))

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

        size = min(w, h)
        # only large enough
        if size > 12:
          bbox = [x, y, w, h]
          bboxes.append(bbox)

      # # for debug
      # img = cv2.imread(img_path)
      # for bbox in bboxes:
      #   x, y, w, h = bbox
      #   cv2.rectangle(img, (int(x), int(y)), (int(x+w), int(y+h)), (0,0,255), 1)
      # cv2.imshow('img', img)
      # cv2.waitKey(0)

      if len(bboxes) > 0:
        bboxes = np.asarray(bboxes, dtype=np.float32)
        bboxes[:, 2] += bboxes[:, 0]
        bboxes[:, 3] += bboxes[:, 1]
        result.append([img_path, bboxes])
    fin.close()
    return result

  txt_dir = os.path.join(cfg.WIDER_DIR, 'wider_face_split')
  train_data = gen(os.path.join(txt_dir, 'wider_face_train.txt'), train_mapper)
  val_data = gen(os.path.join(txt_dir, 'wider_face_val.txt'), val_mapper)
  return (train_data, val_data)


def load_celeba():
  """load celeba dataset and crop the face bbox
  notice: the face bbox may out of the image range
  data: [img_path, bbox, landmark]
  bbox: [x1, y1, x2, y2]
  landmark: [x1, y1, x2, y2, x3, y3, x4, y4, x5, y5], align to top left of the image
  """
  text = os.path.join(cfg.CelebA_DIR, 'list_landmarks_celeba.txt')
  fin = open(text, 'r')
  n = int(fin.readline().strip())
  fin.readline()  # drop

  result = []
  for i in range(n):
    line = fin.readline().strip()
    components = line.split()
    img_path = os.path.join(cfg.CelebA_DIR, 'img_celeba', components[0])
    landmark = np.asarray([int(_) for _ in components[1:]], dtype=np.float32)
    landmark = landmark.reshape((-1, 2)) # 5x2

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
    bbox = [x_new, y_new, x_new + w_new, y_new + h_new]
    bbox = [int(_) for _ in bbox]

    # # for debug
    # img = cv2.imread(img_path)
    # x, y, w, h = bbox
    # cv2.rectangle(img, (x, y), (x+w, y+h), (0,0,255), 1)
    # for j in range(5):
    #   cv2.circle(img, (int(landmark[j, 0]), int(landmark[j, 1])), 2, (0,255,0), -1)
    # cv2.imshow('img', img)
    # cv2.waitKey(0)

    # # normalize landmark
    # landmark[:, 0] = (landmark[:, 0] - bbox[0]) / w_new
    # landmark[:, 1] = (landmark[:, 1] - bbox[1]) / h_new

    landmark = landmark.reshape(-1)
    result.append([img_path, bbox, landmark])

  fin.close()
  ratio = 0.8
  train_n = int(len(result) * ratio)
  train = result[:train_n]
  val = result[train_n:]
  return train, val


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


def crop_face(img, bbox, wrap=True):
  height, width = img.shape[:-1]
  x1, y1, x2, y2 = bbox
  x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
  if x1 >= width or y1 >= height or x2 <= 0 or y2 <= 0:
    print '[WARN] ridiculous x1, y1, x2, y2'
    return None
  if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
    # out of boundary, still crop the face
    if not wrap:
      return None
    h, w = y2 - y1, x2 - x1
    patch = np.zeros((h, w, 3), dtype=np.uint8)
    vx1 = 0 if x1 < 0 else x1
    vy1 = 0 if y1 < 0 else y1
    vx2 = width if x2 > width else x2
    vy2 = height if y2 > height else y2
    sx = -x1 if x1 < 0 else 0
    sy = -y1 if y1 < 0 else 0
    vw = vx2 - vx1
    vh = vy2 - vy1
    patch[sy:sy+vh, sx:sx+vw] = img[vy1:vy2, vx1:vx2]
    return patch
  return img[y1:y2, x1:x2]


class Timer:

  def __init__(self):
    self.start_time = 0
    self.total_time = 0

  def tic(self):
    self.start_time = time.time()

  def toc(self):
    self.total_time = time.time() - self.start_time

  def elapsed(self):
    return self.total_time


if __name__ == '__main__':
  img = cv2.imread('test.jpg')
  bbox = [-100, -200, 300, 400]
  patch = crop_face(img, bbox)
  cv2.imshow('patch', patch)
  cv2.waitKey(0)
