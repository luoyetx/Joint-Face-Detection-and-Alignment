# pylint: disable=no-member
import mxnet as mx


def pnet():
    data = mx.sym.Variable('data')
    conv1 = mx.sym.Convolution(data=data, kernel=(3, 3), num_filter=10, name='conv1')
    prelu1 = mx.sym.LeakyReLU(data=conv1, act_type='prelu', name='prelu1')
    pool1 = mx.sym.Pooling(data=prelu1, kernel=(2, 2), stride=(2, 2), pool_type='max', \
                           pooling_convention='full', name='pool1')
    conv2 = mx.sym.Convolution(data=pool1, kernel=(3, 3), num_filter=16, name='conv2')
    prelu2 = mx.sym.LeakyReLU(data=conv2, act_type='prelu', name='prelu2')
    conv3 = mx.sym.Convolution(data=prelu2, kernel=(3, 3), num_filter=32, name='conv3')
    prelu3 = mx.sym.LeakyReLU(data=conv3, act_type='prelu', name='prelu3')
    score = mx.sym.Convolution(data=prelu3, kernel=(1, 1), num_filter=2, name='score')
    prob = mx.sym.SoftmaxActivation(data=score, mode='channel', name='prob')
    bbox_pred = mx.sym.Convolution(data=prelu3, kernel=(1, 1), num_filter=4, name='bbox_pred')
    landmark_pred = mx.sym.Convolution(data=prelu3, kernel=(1, 1), num_filter=10, name='landmark_pred')
    out = mx.sym.Group([prob, bbox_pred, landmark_pred])
    return out


def rnet():
    data = mx.sym.Variable('data')
    conv1 = mx.sym.Convolution(data=data, kernel=(3, 3), num_filter=28, name='conv1')
    prelu1 = mx.sym.LeakyReLU(data=conv1, act_type='prelu', name='prelu1')
    pool1 = mx.sym.Pooling(data=prelu1, kernel=(3, 3), stride=(2, 2), pool_type='max', \
                           pooling_convention='full', name='pool1')
    conv2 = mx.sym.Convolution(data=pool1, kernel=(3, 3), num_filter=48, name='conv2')
    prelu2 = mx.sym.LeakyReLU(data=conv2, act_type='prelu', name='prelu2')
    pool2 = mx.sym.Pooling(data=prelu2, kernel=(3, 3), stride=(2, 2), pool_type='max', \
                           pooling_convention='full', name='pool2')
    conv3 = mx.sym.Convolution(data=pool2, kernel=(2, 2), num_filter=64, name='conv3')
    prelu3 = mx.sym.LeakyReLU(data=conv3, act_type='prelu', name='prelu3')
    fc = mx.sym.FullyConnected(data=prelu3, num_hidden=128, name='fc')
    prelu4 = mx.sym.LeakyReLU(data=fc, act_type='prelu', name='prelu4')
    score = mx.sym.FullyConnected(data=prelu4, num_hidden=2, name='score')
    prob = mx.sym.SoftmaxActivation(data=score, name='prob')
    bbox_pred = mx.sym.FullyConnected(data=prelu4, num_hidden=4, name='bbox_pred')
    landmark_pred = mx.sym.FullyConnected(data=prelu4, num_hidden=10, name='landmark_pred')
    out = mx.sym.Group([prob, bbox_pred, landmark_pred])
    return out


def onet():
    data = mx.sym.Variable('data')
    conv1 = mx.sym.Convolution(data=data, kernel=(3, 3), num_filter=32, name='conv1')
    prelu1 = mx.sym.LeakyReLU(data=conv1, act_type='prelu', name='prelu1')
    pool1 = mx.sym.Pooling(data=prelu1, kernel=(3, 3), stride=(2, 2), pool_type='max', \
                           pooling_convention='full', name='pool1')
    conv2 = mx.sym.Convolution(data=pool1, kernel=(3, 3), num_filter=64, name='conv2')
    prelu2 = mx.sym.LeakyReLU(data=conv2, act_type='prelu', name='prelu2')
    pool2 = mx.sym.Pooling(data=prelu2, kernel=(3, 3), stride=(2, 2), pool_type='max', \
                           pooling_convention='full', name='pool2')
    conv3 = mx.sym.Convolution(data=pool2, kernel=(3, 3), num_filter=64, name='conv3')
    prelu3 = mx.sym.LeakyReLU(data=conv3, act_type='prelu', name='prelu3')
    pool3 = mx.sym.Pooling(data=prelu3, kernel=(2, 2), stride=(2, 2), pool_type='max', \
                           pooling_convention='full', name='pool3')
    conv4 = mx.sym.Convolution(data=pool3, kernel=(2, 2), num_filter=128, name='conv4')
    prelu4 = mx.sym.LeakyReLU(data=conv4, act_type='prelu', name='prelu4')
    fc = mx.sym.FullyConnected(data=prelu4, num_hidden=256, name='fc')
    prelu5 = mx.sym.LeakyReLU(data=fc, act_type='prelu', name='prelu5')
    score = mx.sym.FullyConnected(data=prelu5, num_hidden=2, name='score')
    prob = mx.sym.SoftmaxActivation(data=score, name='prob')
    bbox_pred = mx.sym.FullyConnected(data=prelu5, num_hidden=4, name='bbox_pred')
    landmark_pred = mx.sym.FullyConnected(data=prelu5, num_hidden=10, name='landmark_pred')
    out = mx.sym.Group([prob, bbox_pred, landmark_pred])
    return out


def lnet():
    data = mx.sym.Variable('data')
    conv1 = mx.sym.Convolution(data=data, kernel=(3, 3), num_filter=50, num_group=5, name='conv1')
    prelu1 = mx.sym.LeakyReLU(data=conv1, act_type='prelu', name='prelu1')
    pool1 = mx.sym.Pooling(data=prelu1, kernel=(3, 3), stride=(2, 2), pool_type='max', \
                           pooling_convention='full', name='pool1')
    conv2 = mx.sym.Convolution(data=pool1, kernel=(3, 3), num_filter=100, num_group=5, name='conv2')
    prelu2 = mx.sym.LeakyReLU(data=conv2, act_type='prelu', name='prelu2')
    pool2 = mx.sym.Pooling(data=prelu2, kernel=(3, 3), stride=(2, 2), pool_type='max', \
                           pooling_convention='full', name='pool2')
    conv3 = mx.sym.Convolution(data=pool2, kernel=(2, 2), num_filter=200, num_group=5, name='conv3')
    prelu3 = mx.sym.LeakyReLU(data=conv3, act_type='prelu', name='prelu3')
    conv4 = mx.sym.Convolution(data=prelu3, kernel=(3, 3), stride=(3, 3), num_filter=100, num_group=5, name='conv4')
    prelu4 = mx.sym.LeakyReLU(data=conv4, act_type='prelu', name='prelu4')
    conv5 = mx.sym.Convolution(data=prelu4, kernel=(1, 1), num_filter=50, num_group=5, name='conv5')
    prelu5 = mx.sym.LeakyReLU(data=conv5, act_type='prelu', name='prelu5')
    conv6 = mx.sym.Convolution(data=prelu5, kernel=(1, 1), num_filter=10, num_group=5, name='conv6')
    out = mx.sym.Reshape(conv6, shape=(-1, 10))
    return out


if __name__ == '__main__':
    p = pnet()
    r = rnet()
    o = onet()
    l = lnet()
    mx.viz.plot_network(p, shape={'data': (1, 3, 12, 12)}).render('tmp/pnet')
    mx.viz.plot_network(r, shape={'data': (1, 3, 24, 24)}).render('tmp/rnet')
    mx.viz.plot_network(o, shape={'data': (1, 3, 48, 48)}).render('tmp/onet')
    mx.viz.plot_network(l, shape={'data': (1, 15, 24, 24)}).render('tmp/lnet')
