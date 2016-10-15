#!/usr/bin/env python2.7

import os
import cv2
from jfda.utils import Timer
from jfda.detector import JfdaDetector


def main(args):
  net = ['proto/p.prototxt', 'model/p.caffemodel',
         'proto/r.prototxt', 'model/r.caffemodel',
         'proto/o.prototxt', 'model/o.caffemodel',]
  detector = JfdaDetector(net)
  param = {
    'ths': [0.6, 0.7, 0.8],
    'factor': 0.7,
    'min_size': 24,
  }
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
      timer.tic()
      bb, ts = detector.detect(img, debug=True, **param)
      timer.toc()
      print 'detect %s costs %.04lfs, s1: %.04lfs, s2: %.04lfs, s3: %.04lfs'%(fp,
            timer.elapsed(), ts[0], ts[1], ts[2])
      out1 = 'tmp/%s_stage1.jpg'%fn
      out2 = 'tmp/%s_stage2.jpg'%fn
      out3 = 'tmp/%s_stage3.jpg'%fn
      gen(img.copy(), bb[0], out1)
      gen(img.copy(), bb[1], out2)
      gen(img.copy(), bb[2], out3)


if __name__ == '__main__':
  args = None
  main(args)
