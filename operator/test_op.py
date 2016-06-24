# coding = utf-8

import numpy as np
import mxnet as mx


def main():
  bbox = mx.sym.Variable('bbox')
  bbox_gt = mx.sym.Variable('bbox_gt')
  landmark = mx.sym.Variable('landmark')
  landmark_gt = mx.sym.Variable('landmark_gt')
  bbox_mask = mx.sym.Variable('bbox_mask')
  landmark_mask = mx.sym.Variable('landmark_mask')
  bbox_gt_refined = mx.sym.MaskIdentity(data=bbox, label=bbox_gt, mask=bbox_mask)
  landmark_gt_refined = mx.sym.MaskIdentity(data=landmark, label=landmark_gt, mask=landmark_mask)
  output = mx.sym.Group([bbox_gt_refined, landmark_gt_refined])

  bbox_a = mx.nd.array(np.array([[1, 2, 3, 4],
                                 [5, 6, 7, 8],
                                 [9, 7, 8, 6]]))
  bbox_gt_a = mx.nd.array(np.array([[4, 3, 2, 1],
                                    [0, 0, 0, 0],
                                    [0, 0, 0, 0]]))
  bbox_mask_a = mx.nd.array(np.array([[0],
                                      [1],
                                      [1]]))
  landmark_a = mx.nd.array(np.array([[1, 2, 3, 4],
                                     [8, 6, 5, 7]]))
  landmark_gt_a = mx.nd.array(np.array([[4, 3, 2, 1],
                                        [0, 0, 0, 0]]))
  landmark_mask_a = mx.nd.array(np.array([[0],
                                          [1]]))
  args = {'bbox': bbox_a,
          'bbox_gt': bbox_gt_a,
          'landmark': landmark_a,
          'landmark_gt': landmark_gt_a,
          'bbox_mask': bbox_mask_a,
          'landmark_mask': landmark_mask_a,}
  # test on cpu
  print 'test on cpu'
  executor = output.bind(ctx=mx.cpu(),
                         args=args)
  executor.forward()
  output_a = executor.outputs
  bbox_gt_refined_a, landmark_gt_refined_a = output_a[0], output_a[1]
  print 'bbox_gt'
  print bbox_gt_a.asnumpy()
  print 'bbox_gt_refined'
  print bbox_gt_refined_a.asnumpy()
  print 'landmark_gt'
  print landmark_gt_a.asnumpy()
  print 'landmark_gt_refined'
  print landmark_gt_refined_a.asnumpy()
  # test on gpu
  print 'test on gpu'
  executor = output.bind(ctx=mx.gpu(0),
                         args=args)
  executor.forward()
  output_a = executor.outputs
  bbox_gt_refined_a, landmark_gt_refined_a = output_a[0], output_a[1]
  print 'bbox_gt'
  print bbox_gt_a.asnumpy()
  print 'bbox_gt_refined'
  print bbox_gt_refined_a.asnumpy()
  print 'landmark_gt'
  print landmark_gt_a.asnumpy()
  print 'landmark_gt_refined'
  print landmark_gt_refined_a.asnumpy()


if __name__ == '__main__':
  main()
