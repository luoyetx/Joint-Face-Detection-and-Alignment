#!/usr/bin/env bash

GPU=0

# pnet
echo "Generate Data for pNet"
python jfda/prepare.py --net p --wider --celeba --worker 8
echo "Train pNet"
python jfda/train.py --net p --gpu $GPU --size 128 --lr 0.05 --lrw 0.1 --lrp 5 --epoch 20

# rnet
echo "Generate Data for rNet"
python jfda/prepare.py --net r --celeba --worker 8
python jfda/prepare.py --net r --gpu $GPU --detect --wider --worker 4
echo "Train rNet"
python jfda/train.py --net r --gpu $GPU --size 64 --lr 0.01 --lrw 0.1 --lrp 10 --epoch 40

# onet
echo "Generate Data for oNet"
python jfda/prepare.py --net o --celeba --worker 8
python jfda/prepare.py --net o --gpu $GPU --detect --wider --worker 2
echo "Train oNet"
python jfda/train.py --net o --gpu $GPU --size 64 --lr 0.01 --lrw 0.1 --lrp 10 --epoch 40

# lnet
echo "Generate Data for lNet"
python jfda/lnet.py --prepare --worker 8
echo "Train lNet"
python jfda/lnet.py --train --gpu $GPU --lr 0.1 --lrw 0.1 --lrp 2 --epoch 10
