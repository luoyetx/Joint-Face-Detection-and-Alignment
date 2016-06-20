#!/usr/bin/env python2.7
# coding = utf-8

import os
import multiprocessing
import mxnet as mx
from utils import load_wider, load_celeba
from utils import calc_IoU, check_bbox
from utils import get_logger


qs = [multiprocessing.Queue(1024) for i in range(2)]


def write_list(path_out, image_list):
  with open(path_out, 'w') as fout:
    for i in xrange(len(image_list)):
      line = '%d\t'%image_list[i][-1]
      for j in image_list[i][1:-1]:
          line += '%f\t'%j
      line += '%s\n'%image_list[i][0]
      fout.write(line)


def write_work(q, prefix):
  logger = get_logger()
  sink = []
  idx = 0
  record = mx.recordio.MXRecordIO(prefix+'.rec', 'w')
  while True:
    stat, s, item = q.get()
    if stat == 'finish':
      write_list(prefix+'.lst', sink)
      break
    record.write(s)
    item.append(idx)
    sink.append(item)
    idx += 1
    if len(sink) % 1000 == 0:
      logger.info('count: %d', len(sink))


def main():
  pass


if __name__ == '__main__':
  main()
