#!/usr/bin/env python2.7

import os
import cv2
from jfda.utils import Timer
from jfda.detector import JfdaDetector


def main(args):
  net1 = ['proto/p.prototxt', 'model/p.caffemodel',]
  net2 = ['proto/p.prototxt', 'model/p.caffemodel',
          'proto/r.prototxt', 'model/r.caffemodel',]
  net3 = ['proto/p.prototxt', 'model/p.caffemodel',
          'proto/r.prototxt', 'model/r.caffemodel',
          'proto/o.prototxt', 'model/o.caffemodel',]
  detector1 = JfdaDetector(net1)
  detector2 = JfdaDetector(net2)
  detector3 = JfdaDetector(net3)
  ths = [0.6, 0.8, 0.9]
  min_size = 24
  factor = 0.7
  timer = Timer()

  def gen(img, bboxes, outname):
    for i in range(len(bboxes)):
      x1, y1, x2, y2, score = bboxes[i, :5]
      x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
      cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
      cv2.putText(img, '%.03f'%score, (x1, y1), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
      # landmark
      landmark = bboxes[i, 9:].reshape((5, 2))
      for j in range(5):
        x, y = landmark[j]
        x, y = int(x), int(y)
        cv2.circle(img, (x, y), 2, (0, 255, 0), -1)
    cv2.imwrite(outname, img)

  with open('demo.txt', 'r') as fin:
    for line in fin.readlines():
      fp = line.strip()
      fn = os.path.basename(fp).split('.')[0]
      img = cv2.imread(fp, cv2.IMREAD_COLOR)
      bboxes1 = detector1.detect(img, ths, min_size, factor)
      bboxes2 = detector2.detect(img, ths, min_size, factor)
      timer.tic()
      bboxes3 = detector3.detect(img, ths, min_size, factor)
      timer.toc()
      print 'detect %s costs %.04f s'%(fp, timer.elapsed())
      out1 = 'tmp/%s_stage1.jpg'%fn
      out2 = 'tmp/%s_stage2.jpg'%fn
      out3 = 'tmp/%s_stage3.jpg'%fn
      gen(img.copy(), bboxes1, out1)
      gen(img.copy(), bboxes2, out2)
      gen(img.copy(), bboxes3, out3)


if __name__ == '__main__':
  args = None
  main(args)
