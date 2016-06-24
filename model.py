#!/usr/bin/env python2.7
# coding = utf-8

import mxnet as mx


def get_model(net='p', is_train=True):
  """get model for P-Net, R-Net, O-Net
  """
  data = mx.sym.Variable('data')
  if net == 'p':
    conv1 = mx.sym.Convolution(data=data, kernel=(3, 3), num_filter=10, name='conv1')
    relu1 = mx.sym.Activation(data=conv1, act_type='relu', name='relu1')
    pool1 = mx.sym.Pooling(data=relu1, pool_type='max', kernel=(3, 3), stride=(2, 2), name='pool1')
    conv2 = mx.sym.Convolution(data=pool1, kernel=(3, 3), num_filter=16, name='conv2')
    relu2 = mx.sym.Activation(data=conv2, act_type='relu', name='relu2')
    conv3 = mx.sym.Convolution(data=relu2, kernel=(3, 3), num_filter=32, name='conv3')
    relu3 = mx.sym.Activation(data=conv3, act_type='relu', name='relu3')
    # output
    if is_train:
      # during training, reshape Kx1x1 to K
      face_ = mx.sym.Convolution(data=relu3, kernel=(1, 1), num_filter=2, name='face_')
      bbox_ = mx.sym.Convolution(data=relu3, kernel=(1, 1), num_filter=4, name='bbox_')
      landmark_ = mx.sym.Convolution(data=relu3, kernel=(1, 1), num_filter=10, name='landmark_')
      face = mx.sym.Flatten(data=face_, name='face')
      bbox = mx.sym.Flatten(data=bbox_, name='bbox')
      landmark = mx.sym.Flatten(data=landmark_, name='landmark')
    else:
      face = mx.sym.FullyConnected(data=relu4, num_hidden=2, name='face')
      bbox = mx.sym.FullyConnected(data=relu4, num_hidden=4, name='bbox')
      landmark = mx.sym.FullyConnected(data=relu4, num_hidden=10, name='landmark')
  elif net == 'r':
    conv1 = mx.sym.Convolution(data=data, kernel=(3, 3), num_filter=28, name='conv1')
    relu1 = mx.sym.Activation(data=conv1, act_type='relu', name='relu1')
    pool1 = mx.sym.Pooling(data=relu1, pool_type='max', kernel=(3, 3), stride=(2, 2), name='pool1')
    conv2 = mx.sym.Convolution(data=pool1, kernel=(3, 3), num_filter=48, name='conv2')
    relu2 = mx.sym.Activation(data=conv2, act_type='relu', name='relu2')
    pool2 = mx.sym.Pooling(data=relu2, pool_type='max', kernel=(3, 3), stride=(2, 2), name='pool2')
    conv3 = mx.sym.Convolution(data=pool2, kernel=(2, 2), num_filter=64, name='conv3')
    relu3 = mx.sym.Activation(data=conv3, act_type='relu', name='relu3')
    flatten = mx.sym.Flatten(data=relu3, name='flatten')
    fc = mx.sym.FullyConnected(data=flatten, num_hidden=128, name='fc')
    relu4 = mx.sym.Activation(data=fc, act_type='relu', name='relu4')
    # output
    face = mx.sym.FullyConnected(data=relu4, num_hidden=2, name='face')
    bbox = mx.sym.FullyConnected(data=relu4, num_hidden=4, name='bbox')
    landmark = mx.sym.FullyConnected(data=relu4, num_hidden=10, name='landmark')
  elif net == 'o':
    conv1 = mx.sym.Convolution(data=data, kernel=(3, 3), num_filter=32, name='conv1')
    relu1 = mx.sym.Activation(data=conv1, act_type='relu', name='relu1')
    pool1 = mx.sym.Pooling(data=relu1, pool_type='max', kernel=(3, 3), stride=(2, 2), name='pool1')
    conv2 = mx.sym.Convolution(data=pool1, kernel=(3, 3), num_filter=64, name='conv2')
    relu2 = mx.sym.Activation(data=conv2, act_type='relu', name='relu2')
    pool2 = mx.sym.Pooling(data=relu2, pool_type='max', kernel=(3, 3), stride=(2, 2), name='pool2')
    conv3 = mx.sym.Convolution(data=pool2, kernel=(3, 3), num_filter=64, name='conv3')
    relu3 = mx.sym.Activation(data=conv3, act_type='relu', name='relu3')
    pool3 = mx.sym.Pooling(data=relu3, pool_type='max', kernel=(2, 2), stride=(2, 2), name='pool3')
    conv4 = mx.sym.Convolution(data=pool3, kernel=(2, 2), num_filter=128, name='conv4')
    relu4 = mx.sym.Activation(data=conv4, act_type='relu', name='relu4')
    flatten = mx.sym.Flatten(data=relu4, name='flatten')
    fc = mx.sym.FullyConnected(data=flatten, num_hidden=256, name='fc')
    relu5 = mx.sym.Activation(data=fc, act_type='relu', name='relu5')
    # output
    face = mx.sym.FullyConnected(data=relu5, num_hidden=2, name='face')
    bbox = mx.sym.FullyConnected(data=relu5, num_hidden=4, name='bbox')
    landmark = mx.sym.FullyConnected(data=relu5, num_hidden=10, name='landmark')
  else:
    raise RuntimeError('No network type %s'%net)
  output = mx.sym.Group([face, bbox, landmark])
  return output


def add_loss(model, ws=[1.0, 1.0, 1.0]):
  """add loss
  """
  face_gt = mx.sym.Variable('face_gt')
  bbox_gt = mx.sym.Variable('bbox_gt')
  landmark_gt = mx.sym.Variable('landmark_gt')
  bbox_mask = mx.sym.Variable('bbox_mask')
  landmark_mask = mx.sym.Variable('landmark_mask')
  face, bbox, landmark = model[0], model[1], model[2]

  # bbox_gt_ref = mx.sym.MaskIdentity(*[bbox, bbox_gt, bbox_mask], name='bbox_gt_ref')
  # landmark_gt_ref = mx.sym.MaskIdentity(*[landmark, landmark_gt, landmark_mask], name='landmark_gt_ref')
  bbox_gt_ref = mx.sym.MaskIdentity(data=bbox, label=bbox_gt,
                                    mask=bbox_mask, name='bbox_gt_ref')
  landmark_gt_ref = mx.sym.MaskIdentity(data=landmark, label=landmark_gt,
                                        mask=landmark_mask, name='landmark_gt_ref')
  output1 = mx.sym.SoftmaxOutput(data=face, label=face_gt, grad_scale=ws[0], name='output1')
  output2 = mx.sym.LinearRegressionOutput(data=bbox, label=bbox_gt_ref, grad_scale=ws[1], name='output2')
  output3 = mx.sym.LinearRegressionOutput(data=landmark, label=landmark_gt_ref, grad_scale=ws[2], name='output3')
  output = mx.sym.Group([output1, output2, output3])
  return output


if __name__ == '__main__':
  print 'get pnet'
  p = get_model('p', is_train=True)
  print 'get loss to pnet'
  pl = add_loss(p, ws=[1.0, 0.5, 0.5])

  print 'draw symbols'
  shape = {
    'data': (1, 3, 12, 12),
    'face_gt': (1,),
    'bbox_gt': (1, 4),
    'landmark_gt': (1, 10),
    'bbox_mask': (1, 1),
    'landmark_mask': (1, 1),
  }
  print 'draw pnet'
  fig = mx.viz.plot_network(p, shape={'data': (1, 3, 12, 12)})
  fig.render('pnet')
  print 'draw pnet with loss'
  fig = mx.viz.plot_network(pl, shape=shape)
  fig.render('pnet_with_loss')

  print 'draw rnet'
  r = get_model('r', is_train=True)
  rl = add_loss(r, ws=[1.0, 0.5, 0.5])
  fig = mx.viz.plot_network(r, shape={'data': (1, 3, 24, 24)})
  fig.render('rnet')
  print 'draw rnet with loss'
  shape['data'] = (1, 3, 24, 24)
  fig = mx.viz.plot_network(rl, shape=shape)
  fig.render('rnet_with_loss')

  print 'draw onet'
  o = get_model('o', is_train=False)
  ol = add_loss(o, ws=[1.0, 0.5, 1.0])
  fig = mx.viz.plot_network(o, shape={'data': (1, 3, 48, 48)})
  fig.render('onet')
  print 'draw onet with loss'
  shape['data'] = (1, 3, 48, 48)
  fig = mx.viz.plot_network(ol, shape=shape)
  fig.render('onet_with_loss')

