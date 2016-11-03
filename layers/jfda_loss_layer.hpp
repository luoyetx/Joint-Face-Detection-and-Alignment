#ifndef CAFFE_JFDA_LOSS_LAYER_HPP_
#define CAFFE_JFDA_LOSS_LAYER_HPP_

#include <vector>
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/loss_layer.hpp"

namespace caffe {

// Howto
// layer {
//  name: "loss"
//  type: "JfdaLoss"
//  bottom: "score"
//  bottom: "bbox_pred"
//  bottom: "landmark_pred"
//  bottom: "bbox_target"
//  bottom: "landmark_target"
//  bottom: "label"
//  top: "face_cls_loss"
//  top: "bbox_reg_loss"
//  top: "landmark_reg_loss"
//  top: "face_cls_pos_acc"
//  top: "face_cls_neg_acc"
//  loss_weight: 1    # face_cls_loss
//  loss_weight: 0.5  # bbox_reg_loss
//  loss_weight: 0.5  # landmark_reg_loss
//  loss_weight: 0    # no loss for neg acc
//  loss_weight: 0    # no loss for pos acc
// }

template<typename Dtype>
class JfdaLossLayer : public Layer<Dtype> {
 public:
  explicit JfdaLossLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}

  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "JfdaLoss"; }

  virtual inline int ExactNumBottomBlobs() const { return 6; }
  virtual inline int ExactNumTopBlobs() const { return 5; }
  virtual inline bool AutoTopBlobs() const { return true; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

 public:
  float drop_loss_rate_;
  Blob<Dtype> mask_;
  Blob<Dtype> prob_;
  Blob<Dtype> bbox_diff_;
  Blob<Dtype> landmark_diff_;
};

}  // namespace caffe

#endif  // CAFFE_JFDA_LOSS_LAYER_HPP_
