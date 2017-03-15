Profile
=======

- profile-big-lnet-unoptim-caffe.json 使用 5 个分支的 LNet 网络加上未经优化过的 Mini-Caffe，151 ms
- profile-big-lnet-optim-caffe.json 使用 5 个分支的 LNet 网络加上经过优化的 Mini-Caffe，89 ms
- profile-small-lnet-unoptim-caffe.json 使用单个较小的 LNet 网络加上未经优化的 Mini-Caffe，130 ms
- profile-small-lnet-optim-caffe.json 使用dange较小的 LNet 网络加上经过优化的 Mini-Caffe，79 ms

主要针对 Mini-Caffe 优化了 CPU 下的 PReLU Layer 和 Pooling Layer

- stage1:PReLU 从 9.375 ms 降到 1.564 ms
- stage2:PReLU 从 10.179 ms 降到 2.029 ms
- stage3:PReLU 从 17.361 ms 降到 2.380 ms
- stage4:PReLU 从 3.304 ms 降到 0.489 ms
- stage1:Pooling 从 2.466 ms 降到 1.662 ms
- stage2:Pooling 从 12.431 ms 降到 6.461 ms
- stage3:Pooling 从 15.170 ms 降到 9.746 ms
- stage4:Pooling 从 4.007 ms 降到 1.765 ms
