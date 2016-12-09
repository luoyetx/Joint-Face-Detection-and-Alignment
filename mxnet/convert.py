#!/usr/bin/env python2.7

import argparse
import caffe
import mxnet as mx
import numpy as np
from model import pnet, rnet, onet, lnet


def get_params(caffe_param):
    """Get all params from caffe, layer type is Convolution, PReLU, InnerProduct
    """
    arg_params = {}
    for k, v in caffe_param.iteritems():
        if 'conv' in k:
            # Convolution
            arg_params[k+'_weight'] = mx.nd.array(v[0].data)
            arg_params[k+'_bias'] = mx.nd.array(v[1].data)
        elif 'prelu' in k:
            # PReLU
            arg_params[k+'_gamma'] = mx.nd.array(v[0].data)
        else:
            # InnerProduct
            arg_params[k+'_weight'] = mx.nd.array(v[0].data)
            arg_params[k+'_bias'] = mx.nd.array(v[1].data)
    return arg_params


def test_net(mx_net, caffe_net, data):
    """test network
    """
    caffe_net.blobs['data'].reshape(*data.shape)
    caffe_net.blobs['data'].data[...] = data
    caffe_net.forward()
    caffe_prob = caffe_net.blobs['prob'].data
    caffe_bbox = caffe_net.blobs['bbox_pred'].data
    caffe_landmark = caffe_net.blobs['landmark_pred'].data
    batch = mx.io.DataBatch(data=[mx.nd.array(data)], label=None)
    mx_net.forward(batch, is_train=False)
    mx_prob, mx_bbox, mx_landmark = [x.asnumpy() for x in mx_net.get_outputs()]
    mse = lambda x, y: np.square(x-y).mean()
    print 'prob mse:', mse(caffe_prob, mx_prob)
    print 'bbox mse:', mse(caffe_bbox, mx_bbox)
    print 'landmark mse:', mse(caffe_landmark, mx_landmark)


def test_lnet(mx_net, caffe_net, data):
    """test lnet
    """
    caffe_net.blobs['data'].reshape(*data.shape)
    caffe_net.blobs['data'].data[...] = data
    caffe_net.forward()
    caffe_offset = caffe_net.blobs['landmark_offset'].data
    batch = mx.io.DataBatch(data=[mx.nd.array(data)], label=None)
    mx_net.forward(batch, is_train=False)
    mx_offset = mx_net.get_outputs()[0].asnumpy()
    mse = lambda x, y: np.square(x-y).mean()
    print 'landmark offset mse:', mse(caffe_offset, mx_offset)


def convert(net_type, args):
    """Convert a network
    """
    if net_type == 'pnet':
        mx_net = pnet()
        caffe_net = caffe.Net(args.proto_dir + '/p.prototxt', caffe.TEST, weights=args.model_dir + '/p.caffemodel')
        input_channel = 3
        input_size = 12
        mode_prefix = 'tmp/pnet'
    elif net_type == 'rnet':
        mx_net = rnet()
        caffe_net = caffe.Net(args.proto_dir + '/r.prototxt', caffe.TEST, weights=args.model_dir + '/r.caffemodel')
        input_channel = 3
        input_size = 24
        mode_prefix = 'tmp/rnet'
    elif net_type == 'onet':
        mx_net = onet()
        caffe_net = caffe.Net(args.proto_dir + '/o.prototxt', caffe.TEST, weights=args.model_dir + '/o.caffemodel')
        input_channel = 3
        input_size = 48
        mode_prefix = 'tmp/onet'
    elif net_type == 'lnet':
        mx_net = lnet()
        caffe_net = caffe.Net(args.proto_dir + '/l.prototxt', caffe.TEST, weights=args.model_dir + '/l.caffemodel')
        input_channel = 15
        input_size = 24
        mode_prefix = 'tmp/lnet'
    else:
        raise ValueError("No such net type (%s)"%net_type)

    arg_params = get_params(caffe_net.params)
    mx_mod = mx.mod.Module(symbol=mx_net, data_names=('data'), label_names=None)
    mx_mod.bind(data_shapes=[('data', (100, input_channel, input_size, input_size)),])
    mx_mod.set_params(arg_params=arg_params, aux_params=None, allow_missing=True)
    mx.model.save_checkpoint(mode_prefix, 0, mx_net, arg_params, {})

    # test
    data = np.random.rand(100, input_channel, input_size, input_size).astype(np.float32)
    if net_type == 'lnet':
        test_lnet(mx_mod, caffe_net, data)
    else:
        test_net(mx_mod, caffe_net, data)

    return mx_mod, caffe_net


def main(args):
    # pnet
    print 'convert pnet'
    convert('pnet', args)
    # rnet
    print 'convert rnet'
    convert('rnet', args)
    # onet
    print 'convert onet'
    convert('onet', args)
    # lnet
    print 'convert lnet'
    convert('lnet', args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--proto-dir', type=str, default='../proto', help="caffe proto directory")
    parser.add_argument('--model-dir', type=str, default='../model', help="caffe mode directory")
    parser.add_argument('--out-dir', type=str, default='./tmp', help="mxnet output model directory")
    args = parser.parse_args()
    print args
    main(args)
