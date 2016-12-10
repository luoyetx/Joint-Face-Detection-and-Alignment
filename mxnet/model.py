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
    sliced = mx.sym.SliceChannel(data=data, num_outputs=5)
    out = []
    for i in range(1, 6):
        conv1 = mx.sym.Convolution(data=sliced[i-1], kernel=(3, 3), num_filter=28, name='conv1_%d'%i)
        prelu1 = mx.sym.LeakyReLU(data=conv1, act_type='prelu', name='prelu1_%d'%i)
        pool1 = mx.sym.Pooling(data=prelu1, kernel=(3, 3), stride=(2, 2), pool_type='max', \
                               pooling_convention='full', name='pool1_%d'%i)
        conv2 = mx.sym.Convolution(data=pool1, kernel=(3, 3), num_filter=48, name='conv2_%d'%i)
        prelu2 = mx.sym.LeakyReLU(data=conv2, act_type='prelu', name='prelu2_%d'%i)
        pool2 = mx.sym.Pooling(data=prelu2, kernel=(3, 3), stride=(2, 2), pool_type='max', \
                               pooling_convention='full', name='pool2_%d'%i)
        conv3 = mx.sym.Convolution(data=pool2, kernel=(2, 2), num_filter=64, name='conv3_%d'%i)
        prelu3 = mx.sym.LeakyReLU(data=conv3, act_type='prelu', name='prelu3_%d'%i)
        out.append(prelu3)
    concat = mx.sym.Concat(*out, name='concat')
    fc4 = mx.sym.FullyConnected(data=concat, num_hidden=256, name='fc4')
    prelu4 = mx.sym.LeakyReLU(data=fc4, act_type='prelu', name='prelu4')
    out = []
    for i in range(1, 6):
        fc5 = mx.sym.FullyConnected(data=prelu4, num_hidden=64, name='fc5_%d'%i)
        prelu5 = mx.sym.LeakyReLU(data=fc5, act_type='prelu', name='prelu5_%d'%i)
        fc6 = mx.sym.FullyConnected(data=prelu5, num_hidden=2, name='fc6_%d'%i)
        out.append(fc6)
    out = mx.sym.Concat(*out, name='landmark_offset')
    return out


if __name__ == '__main__':
    p = pnet()
    r = rnet()
    o = onet()
    l = lnet()
    mx.viz.plot_network(p, shape={'data': (1, 3, 12, 12)}).render('tmp/pnet')
    mx.viz.plot_network(r, shape={'data': (1, 3, 24, 24)}).render('tmp/rnet')
    mx.viz.plot_network(o, shape={'data': (1, 3, 48, 48)}).render('tmp/onet')
    # mx.viz doesn't support multi-output from an op
    #mx.viz.plot_network(l, shape={'data': (1, 15, 24, 24)}).render('tmp/lnet')
