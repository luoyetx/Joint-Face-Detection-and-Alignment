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
      face_cls_ = mx.sym.Convolution(data=relu3, kernel=(1, 1), num_filter=2, name='face_cls_')
      face_cls = mx.sym.Flatten(data=face_cls_, name='face_cls')
      bbox_rg_ = mx.sym.Convolution(data=relu3, kernel=(1, 1), num_filter=4, name='bbox_rg_')
      bbox_rg = mx.sym.Flatten(data=bbox_rg_, name='bbox_rg')
      landmark_rg_ = mx.sym.Convolution(data=relu3, kernel=(1, 1), num_filter=10, name='landmark_rg_')
      landmark_rg = mx.sym.Flatten(data=landmark_rg_, name='landmark_rg')
    else:
      face_cls = mx.sym.FullyConnected(data=relu4, num_hidden=2, name='face_cls')
      bbox_rg = mx.sym.FullyConnected(data=relu4, num_hidden=4, name='bbox_rg')
      landmark_rg = mx.sym.FullyConnected(data=relu4, num_hidden=10, name='landmark_rg')
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
    face_cls = mx.sym.FullyConnected(data=relu4, num_hidden=2, name='face_cls')
    bbox_rg = mx.sym.FullyConnected(data=relu4, num_hidden=4, name='bbox_rg')
    landmark_rg = mx.sym.FullyConnected(data=relu4, num_hidden=10, name='landmark_rg')
  elif net == 'o':
    conv1 = mx.sym.Convolution(data=data, kernel=(3, 3), num_filter=32, name='conv1')
    relu1 = mx.sym.Activation(data=conv1, act_type='relu', name='relu1')
    pool1 = mx.sym.Pooling(data=relu1, pool_type='max', kernel=(3, 3), stride=(2, 2), name='pool1')
    conv2 = mx.sym.Convolution(data=pool1, kernel=(3, 3), num_filter=64, name='conv2')
    relu2 = mx.sym.Activation(data=conv2, act_type='relu', name='relu2')
    pool2 = mx.sym.Pooling(data=relu2, pool_type='max', name='pool2')
    conv3 = mx.sym.Convolution(data=pool2, kernel=(3, 3), num_filter=64, name='conv3')
    relu3 = mx.sym.Activation(data=conv3, act_type='relu', name='relu3')
    pool3 = mx.sym.Pooling(data=relu3, pool_type='max', name='pool3')
    conv4 = mx.sym.Convolution(data=pool3, kernel=(2, 2), num_filter=128, name='conv4')
    relu4 = mx.sym.Activation(data=conv4, act_type='relu', name='relu4')
    flatten = mx.sym.Flatten(data=relu4, name='flatten')
    fc = mx.sym.FullyConnected(data=flatten, num_hidden=256, name='fc')
    relu5 = mx.sym.Activation(data=fc, act_type='relu', name='relu5')
    # output
    face_cls = mx.sym.FullyConnected(data=relu5, num_hidden=2, name='face_cls')
    bbox_rg = mx.sym.FullyConnected(data=relu5, num_hidden=4, name='bbox_rg')
    landmark_rg = mx.sym.FullyConnected(data=relu5, num_hidden=10, name='landmark_rg')
  else:
    raise RuntimeError('No network type %s'%net)
  output = mx.sym.Group([face_cls, bbox_rg, landmark_rg])
  return output


def add_loss(model, ws=[1.0, 1.0, 1.0]):
  """add loss
  """
  face_cls_gt = mx.sym.Variable('face')
  bbox_rg_gt = mx.sym.Variable('bbox')
  landmark_rg_gt = mx.sym.Variable('landmark')
  bbox_mask = mx.sym.Variable('bbox_mask')
  landmark_mask = mx.sym.Variable('landmark_mask')
  face_cls, bbox_rg, landmark_rg = model[0], model[1], model[2]

  # gt_refined = mx.sym.MaskIdentity(*[bbox_rg, landmark_rg,
  #                                    bbox_rg_gt, landmark_rg_gt,
  #                                    bbox_mask, landmark_mask],
  #                                  name='mask_identity')
  gt_refined = mx.sym.MaskIdentity(bbox=bbox_rg, landmark=landmark_rg,
                                   bbox_gt=bbox_rg_gt, landmark_gt=landmark_rg_gt,
                                   bbox_mask=bbox_mask, landmark_mask=landmark_mask,
                                   name=('gt_refined'))
  print gt_refined[0].name, gt_refined[1].name
  output1 = mx.sym.SoftmaxOutput(data=face_cls, label=face_cls_gt, grad_scale=ws[0], name='output1')
  output2 = mx.sym.LinearRegressionOutput(data=bbox_rg, label=gt_refined[0], grad_scale=ws[1], name='output2')
  output3 = mx.sym.LinearRegressionOutput(data=landmark_rg, label=gt_refined[1], grad_scale=ws[2], name='output3')
  output = mx.sym.Group([output1, output2, output3])
  return output


if __name__ == '__main__':
  print 'get pnet'
  p = get_model('p', is_train=True)
  p.list_arguments()
  print 'get loss to pnet'
  pl = add_loss(p, ws=[1.0, 0.5, 0.5])
  print 'draw symbols'
  shape = {
    'data': (1, 1, 12, 12),
    'face': (1,),
    'bbox': (1, 4),
    'landmark': (1, 10),
    'bbox_mask': (1,),
    'landmark_mask': (1,),
  }
  fig = mx.viz.plot_network(p, shape={'data': (1,1,12,12)})
  fig.render('pnet')
  fig = mx.viz.plot_network(pl)
  fig.render('pnet_with_loss')
