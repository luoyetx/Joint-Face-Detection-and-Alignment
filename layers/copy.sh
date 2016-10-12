#!/usr/bin/env bash

cp layers/jfda_loss_layer.hpp $CAFFE_HOME/include/caffe/layers/jfda_loss_layer.hpp
cp layers/jfda_loss_layer.cpp $CAFFE_HOME/src/caffe/layers/jfda_loss_layer.cpp
cp layers/jfda_loss_layer.cu $CAFFE_HOME/src/caffe/layers/jfda_loss_layer.cu
cp layers/test_jfda_loss_layer.cpp $CAFFE_HOME/src/caffe/test/test_jfda_loss_layer.cpp
