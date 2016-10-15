#!/usr/bin/env bash

# pnet
echo "Generate Data for pNet"
python jfda/prepare.py --net p --wider --celeba --worker 8
echo "Train pNet"
python jfda/train.py --net p --gpu 0 --size 128 --lr 0.01 --lrw 0.1 --lrp 5 --epoch 20

# rnet
echo "Generate Data for rNet"
python jfda/prepare.py --net r --celeba --worker 8
python jfda/prepare.py --net r --gpu 0 --detect --wider --worker 4
echo "Train rNet"
python jfda/train.py --net r --gpu 0 --size 64 --lr 0.01 --lrw 0.1 --lrp 10 --epoch 40

# onet
echo "Generate Data for oNet"
python jfda/prepare.py --net o --celeba --worker 8
python jfda/prepare.py --net o --gpu 0 --detect --wider --worker 1
echo "Train oNet"
python jfda/train.py --net o --gpu 0 --size 64 --lr 0.01 --lrw 0.1 --lrp 10 --epoch 40
