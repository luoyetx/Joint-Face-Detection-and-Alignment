#!/usr/bin/env bash

set -e
GPU=0
GL=$GLOG_minloglevel

# pnet
echo "Generate Data for pNet"
export GLOG_minloglevel=2
python jfda/prepare.py --net p --wider --celeba --worker 8
echo "Train pNet"
export GLOG_minloglevel=$GL
python jfda/train.py --net p --gpu $GPU --size 128 --lr 0.05 --lrw 0.1 --lrp 5 --wd 0.0001 --epoch 25

# rnet
echo "Generate Data for rNet"
export GLOG_minloglevel=2
python jfda/prepare.py --net r --gpu $GPU --detect --celeba --wider --worker 4
echo "Train rNet"
export GLOG_minloglevel=$GL
python jfda/train.py --net r --gpu $GPU --size 128 --lr 0.05 --lrw 0.1 --lrp 5 --wd 0.0001 --epoch 25

# onet
echo "Generate Data for oNet"
export GLOG_minloglevel=2
python jfda/prepare.py --net o --gpu $GPU --detect --celeba --wider --worker 4
echo "Train oNet"
export GLOG_minloglevel=$GL
python jfda/train.py --net o --gpu $GPU --size 64 --lr 0.05 --lrw 0.1 --lrp 7 --wd 0.0001 --epoch 35

# lnet
echo "Generate Data for lNet"
export GLOG_minloglevel=2
python jfda/lnet.py --prepare --worker 8
echo "Train lNet"
export GLOG_minloglevel=$GL
python jfda/lnet.py --train --gpu $GPU --lr 0.1 --lrw 0.1 --lrp 2 --epoch 10
