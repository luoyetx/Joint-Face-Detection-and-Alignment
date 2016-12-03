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
# cfg.PROPOSAL_SCALES = [0.8, 1.0, 1.2] # proposal sliding window scales
# cfg.PROPOSAL_STRIDES = [0.1] # proposal sliding window strides
cfg.POS_PROPOSAL_SCALES = [0.6, 0.8, 1.0, 1.2]
cfg.POS_PROPOSAL_STRIDE = 0.1
cfg.PART_PROPOSAL_SCALES = [0.6, 0.8, 1.0, 1.2]
cfg.PART_PROPOSAL_STRIDE = 0.1
cfg.NEG_PROPOSAL_SCALES = [0.6, 0.8, 1.0, 1.2]
cfg.NEG_PROPOSAL_STRIDE = 0.1
cfg.POS_PER_FACE = 10 # positive face per face region
cfg.PART_PER_FACE = 10 # part face per face region
cfg.NEG_PER_FACE = 10 # non-face per face region
cfg.LANDMARK_PER_FACE = 10 # landmark face per face region
cfg.NEG_FROM_FR_RATIO = 0.5 # ratio of non-face from face region, non-face are from face region or global image random patch
cfg.NEG_FORCE_BALANCE = True # whether to balance the ratio of negatives from face region and global crop, this will reduce the total number of negative samples
cfg.NEG_PER_IMAGE = 256 # non-face per image, set to 128 if you set cfg.NEG_FORCE_BALANCE=False
cfg.NEG_DETECT_PER_IMAGE = 128 # use for rnet and onet
cfg.NEG_PROPOSAL_RATIO = 10 # total proposal size equals to NEG_PER_IMAGE * NEG_PROPOSAL_RATIO
cfg.NEG_MIN_SIZE = 12 # minimum random crop patch size for non-face

# using previous network to generate data
cfg.PROPOSAL_NETS = {
  'p': None,
  'r': ['proto/p.prototxt', 'model/p.caffemodel'],
  'o': ['proto/p.prototxt', 'model/p.caffemodel', 'proto/r.prototxt', 'model/r.caffemodel'],
}
cfg.DETECT_PARAMS = {
  'min_size': 24,
  'ths': [0.5, 0.5, 0.5],
  'factor': 0.709
}

# training data ratio in a minibatch, [negative, positive, part, landmark]
cfg.DATA_RATIO = {
  'p': [3, 1, 1, 2],
  'r': [3, 1, 1, 2],
  'o': [3, 2, 1, 2],
}

# data augment
cfg.GRAY_PROB = 0.1
cfg.FLIP_PROB = 0.5

# lnet
cfg.SAMPLE_RADIUS = 0.1
cfg.DATASIZE_PER_H5 = 100000
cfg.LNET_SAMPLE_PER_FACE = 3
cfg.LNET_FACE_SCALES = [0.3]
