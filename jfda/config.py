import os
from easydict import EasyDict


cfg = EasyDict()

# data directories
cfg.Data_DIR = 'data/'
cfg.CelebA_DIR = 'data/CelebA/'
cfg.WIDER_DIR = 'data/WIDER/'
cfg.FDDB_DIR = 'data/fddb/'

# cnn input size
cfg.NET_TYPE = 'p'
cfg.NET_INPUT_SIZE = {'p': 12, 'r': 24, 'o': 48}

# data prepare
cfg.USE_DETECT = False
cfg.GPU_ID = -1
cfg.WORKER_N = 4

# random seed
cfg.RNG_SEED = 1

# shuffle data buff before save to lmdb
cfg.SHUFFLE_SIZE = 10000

# ratios
cfg.FACE_OVERLAP = 0.65 # (0.65, 1] is positives
cfg.NONFACE_OVERLAP = 0.3 # [0, 0.3] is negatives
cfg.PARTFACE_OVERLAP = 0.4 # (0.4, 0.65] is part faces

# face proposal
cfg.PROPOSAL_SCALES = [0.8, 1.0, 1.2]
cfg.PROPOSAL_STRIDES = [0.1]
cfg.POS_PER_FACE = 10
cfg.PART_PER_FACE = 10
cfg.LANDMARK_PER_FACE = 5
cfg.NEG_PER_IMAGE = 128
cfg.NEG_PROPOSAL_RATIO = 10 # total proposal size equals to NEG_PER_IMAGE * NEG_PROPOSAL_RATIO
cfg.NEG_MIN_SIZE = 12

cfg.PROPOSAL_NETS = {
  'p': None,
  'r': ['proto/p.prototxt', 'model/p.caffemodel'],
  'o': ['proto/p.prototxt', 'model/p.caffemodel', 'proto/r.prototxt', 'model/r.caffemodel'],
}
cfg.DETECT_PARAMS = {
  'min_size': 24,
  'ths': [0.5, 0.5, 0.5],
  'factor': 0.7
}

# data augment
cfg.GRAY_PROB = 0.1
cfg.FLIP_PROB = 0.5
