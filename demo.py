#!/usr/bin/env python2.7

import os
import argparse
import cv2
from jfda.utils import Timer
from jfda.detector import JfdaDetector


def main(args):
  net = ['proto/p.prototxt', 'model/p.caffemodel',
         'proto/r.prototxt', 'model/r.caffemodel',
         'proto/o.prototxt', 'model/o.caffemodel',
         'proto/l.prototxt', 'model/l.caffemodel',]
  detector = JfdaDetector(net)
  if args.pnet_single:
    detector.set_pnet_single_forward(True)
  param = {
    'ths': [0.6, 0.7, 0.8],
    'factor': 0.709,
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
      dn = os.path.dirname(fp)
      fn = os.path.basename(fp).split('.')[0]
      img = cv2.imread(fp, cv2.IMREAD_COLOR)
      timer.tic()
      bb, ts = detector.detect(img, debug=True, **param)
      timer.toc()
      print 'detect %s costs %.04lfs'%(fp, timer.elapsed())
      print 'image size = (%d x %d), s1: %.04lfs, s2: %.04lfs, s3: %.04lfs, s4: %.04lf'%(
            img.shape[0], img.shape[1], ts[0], ts[1], ts[2], ts[3])
      print 'bboxes, s1: %d, s2: %d, s3: %d, s4: %d'%(len(bb[0]), len(bb[1]), len(bb[2]), len(bb[3]))
      out1 = '%s/%s_stage1.jpg'%(dn, fn)
      out2 = '%s/%s_stage2.jpg'%(dn, fn)
      out3 = '%s/%s_stage3.jpg'%(dn, fn)
      out4 = '%s/%s_stage4.jpg'%(dn, fn)
      gen(img.copy(), bb[0], out1)
      gen(img.copy(), bb[1], out2)
      gen(img.copy(), bb[2], out3)
      gen(img.copy(), bb[3], out4)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--gpu', type=int, default=-1, help='gpu id to use, -1 for cpu')
  parser.add_argument('--pnet-single', action='store_true', help='pnet use single forward')
  args = parser.parse_args()

  if args.gpu >= 0:
    import caffe
    caffe.set_mode_gpu()
    caffe.set_device(args.gpu)

  main(args)
