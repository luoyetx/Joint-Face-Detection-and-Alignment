Joint-Face-Detection-and-Alignment
==================================

Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks.

## How to

```
$ export PYTHONPATH=/path/to/Joint-Face-Detection-and-Alignment:$PYTHONPATH
```

## Generate training data

```
$ python data/p.py --net p --face --nonface --landmark
$ python data/p.py --net r --face --nonface --landmark
$ python data/p.py --net o --face --nonface --landmark
```

## References

- [A Convolutional Neural Network Cascade for Face Detection](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Li_A_Convolutional_Neural_2015_CVPR_paper.pdf)
- [Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks](http://arxiv.org/abs/1604.02878)
