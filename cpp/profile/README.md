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

CPU 下 demo.py 的速度测试

利用原始 Caffe 的结果

```
detect img/test1.jpg costs 1.7810s
image size = (1200 x 1920), s1: 1.0680s, s2: 0.4980s, s3: 0.1900s, s4: 0.0180
bboxes, s1: 679, s2: 54, s3: 9, s4: 9
detect img/test2.jpg costs 5.5220s
image size = (1412 x 3000), s1: 2.2010s, s2: 1.3010s, s3: 1.8240s, s4: 0.1880
bboxes, s1: 1886, s2: 496, s3: 114, s4: 114
detect img/test3.jpg costs 0.7170s
image size = (778 x 1024), s1: 0.3630s, s2: 0.2670s, s3: 0.0760s, s4: 0.0060
bboxes, s1: 401, s2: 21, s3: 5, s4: 5
```

利用 MiniCaffe 的结果

```
detect img/test1.jpg costs 1.0250s
image size = (1200 x 1920), s1: 0.6660s, s2: 0.2360s, s3: 0.1100s, s4: 0.0090
bboxes, s1: 679, s2: 54, s3: 9, s4: 9
detect img/test2.jpg costs 3.1460s
image size = (1412 x 3000), s1: 1.3690s, s2: 0.6740s, s3: 1.0030s, s4: 0.0960
bboxes, s1: 1886, s2: 496, s3: 114, s4: 114
detect img/test3.jpg costs 0.3730s
image size = (778 x 1024), s1: 0.1960s, s2: 0.1270s, s3: 0.0400s, s4: 0.0050
bboxes, s1: 401, s2: 21, s3: 5, s4: 5
```

### Update

LNet 利用卷积替换掉全连接，在保证精度不变的情况下，提高了速度。GTX 1070 上从 17.59ms 降到了 13.907ms。
