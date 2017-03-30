#!/usr/bin/env python2.7
# pylint: disable=bad-indentation, no-member, invalid-name, line-too-long

import argparse
import cv2
import caffe
from jfda.config import cfg
from jfda.utils import get_logger, Timer
from jfda.detector import JfdaDetector


def main(args):
  if args.net == 'p':
    net = ['proto/p.prototxt', 'model/p.caffemodel']
  elif args.net == 'r':
    net = ['proto/p.prototxt', 'model/p.caffemodel',
           'proto/r.prototxt', 'model/r.caffemodel']
  else:
    net = ['proto/p.prototxt', 'model/p.caffemodel',
           'proto/r.prototxt', 'model/r.caffemodel',
           'proto/o.prototxt', 'model/o.caffemodel',]
  detector = JfdaDetector(net)
  if args.pnet_single:
    detector.set_pnet_single_forward(True)
  logger = get_logger()
  counter = 0
  timer = Timer()
  param = {
    'ths': [0.5, 0.5, 0.4],
    'factor': 0.709,
    'min_size': 24,
  }
  total_time = 0.
  total_width = 0
  total_height = 0
  for i in range(10):
    logger.info('Process FOLD-%02d', i)
    txt_in = cfg.FDDB_DIR + '/FDDB-folds/FDDB-fold-%02d.txt'%(i + 1)
    txt_out = cfg.FDDB_DIR + '/result/fold-%02d-out.txt'%(i + 1)
    answer_in = cfg.FDDB_DIR + '/FDDB-folds/FDDB-fold-%02d-ellipseList.txt'%(i + 1)
    fin = open(txt_in, 'r')
    fout = open(txt_out, 'w')
    ain = open(answer_in, 'r')
    for line in fin.readlines():
      line = line.strip()
      in_file = cfg.FDDB_DIR + '/images/' + line + '.jpg'
      out_file = cfg.FDDB_DIR + '/result/images/' + line.replace('/', '-') + '.jpg'
      counter += 1
      img = cv2.imread(in_file, cv2.IMREAD_COLOR)
      timer.tic()
      bboxes = detector.detect(img, **param)
      timer.toc()
      total_time += timer.elapsed()
      h, w = img.shape[:-1]
      total_height += h
      total_width += w
      logger.info('Detect %04d th image costs %.04lfs, average %.04lfs, %d x %d', counter,
                  timer.elapsed(), total_time / counter, total_height / counter, total_width / counter)
      fout.write('%s\n'%line)
      fout.write('%d\n'%len(bboxes))
      for bbox in bboxes:
        x1, y1, x2, y2, score = bbox[:5]
        fout.write('%lf %lf %lf %lf %lf\n'%(x1, y1, x2 - x1, y2 - y1, score))
      # draw ground truth
      ain.readline() # remove fname
      n = int(ain.readline().strip())
      for i in range(n):
        line = ain.readline().strip()
        components = [float(_)  for _ in line.split(' ')[:5]]
        major_axis_radius, minor_axis_radius, angle, center_x, center_y = components
        angle = angle / 3.1415926 * 180.
        center_x, center_y = int(center_x), int(center_y)
        major_axis_radius, minor_axis_radius = int(major_axis_radius), int(minor_axis_radius)
        cv2.ellipse(img, (center_x, center_y), (major_axis_radius, minor_axis_radius), angle, 0, 360, (255, 0, 0), 2)
      # draw and save
      for bbox in bboxes:
        x1, y1, x2, y2, score = bbox[:5]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(img, '%.03f'%score, (x1, y1), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
        landmark = bbox[9:].reshape((5, 2))
        for x, y in landmark:
          x, y = int(x), int(y)
          cv2.circle(img, (x, y), 2, (0, 255, 0), -1)
      cv2.imwrite(out_file, img)
    fin.close()
    fout.close()
    ain.close()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--gpu', type=int, default=-1, help='gpu id to use, -1 for cpu')
  parser.add_argument('--net', type=str, default='o', help='net type to do fddb benchmark')
  parser.add_argument('--pnet-single', action='store_true', help='pnet use single forward')
  args = parser.parse_args()
  assert args.net in ['p', 'r', 'o']

  print args

  if args.gpu >= 0:
    caffe.set_mode_gpu()
    caffe.set_device(args.gpu)
  else:
    caffe.set_mode_cpu()

  main(args)
