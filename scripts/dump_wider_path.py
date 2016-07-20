#!/usr/bin/env python2.7

from data.utils import load_wider


def main():
  train, val = load_wider()

  def write(data, out):
    with open(out, 'w') as fout:
      for img_path, bboxes in data:
        fout.write('%s\n'%img_path)

  write(train, 'tmp/wider_train.txt')
  write(val, 'tmp/wider_val.txt')


if __name__ == '__main__':
  main()
