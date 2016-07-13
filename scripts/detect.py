#!/usr/bin/env python2.7
# coding = utf-8

import argparse
import cv2
from jfda import Detector
from data.utils import load_wider


def main(args):
  detector = Detector('proto/p.prototxt', 'result/p.caffemodel')

  def detect(data):
    counter = 0
    det = []
    for img_path, bboxes in data:
      counter += 1
      img = cv2.imread(img_path, cv2.IMREAD_COLOR)
      bboxes_det = detector.detect(img)
      det.append([img_path, bboxes_det])
      if counter%1000 == 0:
        print 'processing', counter
    return det

  train, val = load_wider()
  train_det = detect(train)
  val_det = detect(val)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--level', type=int, default=1, help='how many nets to detect')
  args = parser.parse_args()
  assert args.level in [1, 2, 3]
  main(args)
