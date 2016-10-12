#!/usr/bin/env python2.7

import shutil
import argparse
import multiprocessing
import numpy as np
import caffe
from caffe.proto import caffe_pb2
import google.protobuf as pb2
from jfda.config import cfg
from jfda.minibatch import MiniBatcher


class Solver:

  def __init__(self, solver_prototxt, args):
    net_type = args.net
    self.net_type = net_type
    input_size = cfg.NET_INPUT_SIZE[net_type]
    db_names_train = ['data/%snet_negative_train'%net_type,
                      'data/%snet_positive_train'%net_type,
                      'data/%snet_part_train'%net_type,
                      'data/%snet_landmark_train'%net_type]
    db_names_test = ['data/%snet_negative_val'%net_type,
                     'data/%snet_positive_val'%net_type,
                     'data/%snet_part_val'%net_type,
                     'data/%snet_landmark_val'%net_type]
    base_size = args.size
    ns = [3*base_size, base_size, base_size, 2*base_size]
    # batcher setup
    batcher_train = MiniBatcher(db_names_train, ns, net_type)
    batcher_test = MiniBatcher(db_names_test, ns, net_type)
    # data queue setup
    queue_train = multiprocessing.Queue(32)
    queue_test = multiprocessing.Queue(32)
    batcher_train.set_queue(queue_train)
    batcher_test.set_queue(queue_test)
    # solver parameter setup
    size_train = batcher_train.get_size()
    size_test = batcher_test.get_size()
    iter_train = max([x/y for x, y in zip(size_train, ns)])
    iter_test = max([x/y for x, y in zip(size_test, ns)])
    max_iter = args.epoch * iter_train
    self.final_model = 'tmp/%snet_iter_%d.caffemodel'%(net_type, max_iter)
    solver_param = caffe_pb2.SolverParameter()
    with open(solver_prototxt, 'r') as fin:
      pb2.text_format.Merge(fin.read(), solver_param)
    solver_param.max_iter = max_iter  # max training iterations
    solver_param.snapshot = iter_train  # save after an epoch
    solver_param.test_interval = iter_train
    solver_param.test_iter[0] = iter_test
    solver_param.base_lr = args.lr
    solver_param.gamma = args.lrw
    solver_param.stepsize = args.lrp * iter_train
    tmp_solver_prototxt = 'tmp/%s_solver.prototxt'%net_type
    with open(tmp_solver_prototxt, 'w') as fout:
      fout.write(pb2.text_format.MessageToString(solver_param))
    # solver setup
    self.solver = caffe.SGDSolver(tmp_solver_prototxt)
    # data layer setup
    layer_train = self.solver.net.layers[0]
    layer_test = self.solver.test_nets[0].layers[0]
    layer_train.set_batch_num(ns[0], ns[1], ns[2], ns[3])
    layer_test.set_batch_num(ns[0], ns[1], ns[2], ns[3])
    layer_train.set_data_queue(queue_train)
    layer_test.set_data_queue(queue_test)
    # start batcher
    batcher_train.start()
    batcher_test.start()
    def cleanup():
      batcher_train.terminate()
      batcher_test.terminate()
      batcher_train.join()
      batcher_test.join()
    import atexit
    atexit.register(cleanup)

  def train_model(self, snapshot_model=None):
    self.solver.solve(snapshot_model)
    # copy model
    shutil.copyfile(self.final_model, 'model/%s.caffemodel'%self.net_type)


def init_caffe(cfg):
  np.random.seed(cfg.RNG_SEED)
  caffe.set_random_seed(cfg.RNG_SEED)
  if cfg.GPU_ID < 0:
    caffe.set_mode_cpu()
  else:
    caffe.set_mode_gpu()
    caffe.set_device(cfg.GPU_ID)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--gpu', type=int, default=0, help='gpu id to use, -1 for cpu')
  parser.add_argument('--net', type=str, default='p', help='net type, p, r, o')
  parser.add_argument('--size', type=int, default=128, help='base batch size')
  parser.add_argument('--epoch', type=int, default=20, help='train epoches')
  parser.add_argument('--snapshot', type=str, default=None, help='snapshot model')
  parser.add_argument('--lr', type=float, default=0.1, help='initial learning rate')
  parser.add_argument('--lrw', type=float, default=0.1, help='lr decay rate')
  parser.add_argument('--lrp', type=int, default=2, help='number of epoches to decay the lr')
  args = parser.parse_args()

  print args

  net_type = args.net
  assert net_type in ['p', 'r', 'o'], "net should be 'p', 'r', 'o'"
  cfg.NET_TYPE = net_type
  cfg.GPU_ID = args.gpu
  init_caffe(cfg)

  solver_prototxt = 'proto/%s_solver.prototxt'%net_type
  solver = Solver(solver_prototxt, args)
  solver.train_model(args.snapshot)
