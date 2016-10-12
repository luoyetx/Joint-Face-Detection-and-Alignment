#include <cmath>
#include <vector>
#include <algorithm>
#include <boost/scoped_ptr.hpp>
#include "gtest/gtest.h"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/jfda_loss_layer.hpp"
#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"
#include <iostream>

using boost::scoped_ptr;
using std::cout;
using std::endl;

namespace caffe {

static const int kBaseSize = 16;
static const int kBatchSize = kBaseSize * 7;

template<typename TypeParam>
class JfdaLossLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  JfdaLossLayerTest()
      : score(new Blob<Dtype>(kBatchSize, 2, 1, 1)),
        label(new Blob<Dtype>(kBatchSize, 1, 1, 1)),
        bbox_pred(new Blob<Dtype>(kBatchSize, 4, 1, 1)),
        bbox_target(new Blob<Dtype>(kBatchSize, 4, 1, 1)),
        landmark_pred(new Blob<Dtype>(kBatchSize, 10, 1, 1)),
        landmark_target(new Blob<Dtype>(kBatchSize, 10, 1, 1)),
        blob_top_data1_(new Blob<Dtype>()),
        blob_top_data2_(new Blob<Dtype>()),
        blob_top_data3_(new Blob<Dtype>()),
        blob_top_data4_(new Blob<Dtype>()),
        blob_top_data5_(new Blob<Dtype>()) {
    Caffe::set_random_seed(0);
    FillerParameter filler_param;
    filler_param.set_std(10);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->score);
    filler.Fill(this->bbox_pred);
    filler.Fill(this->bbox_target);
    filler.Fill(this->landmark_pred);
    filler.Fill(this->landmark_target);
    int n1 = 3 * kBaseSize;
    int n2 = kBaseSize;
    int n3 = kBaseSize;
    int n4 = 2 * kBaseSize;
    Dtype* label_data = label->mutable_cpu_data();
    for (int i = 0; i < n1; i++) {
      label_data[i] = 0;
    }
    for (int i = 0; i < n2; i++) {
      label_data[i + n1] = 1;
    }
    for (int i = 0; i < n3; i++) {
      label_data[i + n1 + n2] = 2;
    }
    for (int i = 0; i < n4; i++) {
      label_data[i + n1 + n2 + n3] = 3;
    }
    blob_bottom_vec_.push_back(score);
    blob_bottom_vec_.push_back(bbox_pred);
    blob_bottom_vec_.push_back(landmark_pred);
    blob_bottom_vec_.push_back(bbox_target);
    blob_bottom_vec_.push_back(landmark_target);
    blob_bottom_vec_.push_back(label);
    blob_top_vec_.push_back(blob_top_data1_);
    blob_top_vec_.push_back(blob_top_data2_);
    blob_top_vec_.push_back(blob_top_data3_);
    blob_top_vec_.push_back(blob_top_data4_);
    blob_top_vec_.push_back(blob_top_data5_);
  }

  virtual ~JfdaLossLayerTest() {
    delete score;
    delete label;
    delete bbox_pred;
    delete bbox_target;
    delete landmark_pred;
    delete landmark_target;
    delete blob_top_data1_;
    delete blob_top_data2_;
    delete blob_top_data3_;
    delete blob_top_data4_;
    delete blob_top_data5_;
  }

  Blob<Dtype>* const score;
  Blob<Dtype>* const label;
  Blob<Dtype>* const bbox_pred;
  Blob<Dtype>* const bbox_target;
  Blob<Dtype>* const landmark_pred;
  Blob<Dtype>* const landmark_target;
  Blob<Dtype>* const blob_top_data1_;
  Blob<Dtype>* const blob_top_data2_;
  Blob<Dtype>* const blob_top_data3_;
  Blob<Dtype>* const blob_top_data4_;
  Blob<Dtype>* const blob_top_data5_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};  // class JfdaLossLayerTest

TYPED_TEST_CASE(JfdaLossLayerTest, TestDtypesAndDevices);

TYPED_TEST(JfdaLossLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  scoped_ptr<JfdaLossLayer<Dtype> > layer(
    new JfdaLossLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  const Dtype* top_loss1 = this->blob_top_data1_->cpu_data();
  const Dtype* top_loss2 = this->blob_top_data2_->cpu_data();
  const Dtype* top_loss3 = this->blob_top_data3_->cpu_data();

  cout << "Loss-1 " << top_loss1[0] << endl;
  cout << "Loss-2 " << top_loss2[0] << endl;
  cout << "Loss-3 " << top_loss3[0] << endl;

  // bbox loss
  const Dtype* bpd = this->bbox_pred->cpu_data();
  const Dtype* btd = this->bbox_target->cpu_data();
  Dtype loss = 0;
  for (int i = 3*kBaseSize; i < 5*kBaseSize; i++) {
    int offset = this->bbox_pred->offset(i, 0, 0, 0);
    for (int j = 0; j < 4; j++) {
      Dtype diff = bpd[offset + j] - btd[offset + j];
      loss += diff * diff;
    }
  }
  loss /= 2 * kBatchSize;
  cout << "BBox loss " << loss << endl;
}

TYPED_TEST(JfdaLossLayerTest, TestBackward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  scoped_ptr<JfdaLossLayer<Dtype> > layer(
      new JfdaLossLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // set loss weight
  this->blob_top_data1_->mutable_cpu_diff()[0] = 1;
  this->blob_top_data2_->mutable_cpu_diff()[0] = 0.5;
  this->blob_top_data3_->mutable_cpu_diff()[0] = 0.5;
  vector<bool> propagate_down;
  layer->Backward(this->blob_top_vec_, propagate_down, this->blob_bottom_vec_);

  cout << "========================" << endl;
  cout << "score gradient" << endl;
  const Dtype* score_grad = this->score->cpu_diff();
  for (int i = 0; i < 4*kBaseSize; i++) {
    cout << score_grad[2 * i] << " " << score_grad[2 * i + 1] << endl;
  }
  cout << "========================" << endl;
  cout << "bbox gradient" << endl;
  const Dtype* bbox_grad = this->bbox_pred->cpu_diff();
  for (int i = 3*kBaseSize; i < 5*kBaseSize; i++) {
    for (int j = 0; j < 4; j++) {
      cout << bbox_grad[this->bbox_pred->offset(i, j, 0, 0)] << " ";
    }
    cout << endl;
  }
  cout << "========================" << endl;
  cout << "bbox different" << endl;
  const Dtype* bbox_diff = layer->bbox_diff_.cpu_data();
  for (int i = 3*kBaseSize; i < 5*kBaseSize; i++) {
    for (int j = 0; j < 4; j++) {
      cout << bbox_diff[layer->bbox_diff_.offset(i, j, 0, 0)] << " ";
    }
    cout << endl;
  }

  // gradient check of bbox regression
  cout << "========================" << endl;
  cout << "bbox gradient check" << endl;
  Dtype loss1, loss2, gradient1, gradient2, esp;
  esp = 1e-2;
  int offset = this->bbox_pred->offset(4*kBaseSize, 2);
  gradient1 = this->bbox_pred->cpu_diff()[offset];
  Dtype x = this->bbox_pred->cpu_data()[offset];
  this->bbox_pred->mutable_cpu_data()[offset] = x - esp;
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  loss1 = this->blob_top_data1_->cpu_data()[0] * this->blob_top_data1_->cpu_diff()[0] +
          this->blob_top_data2_->cpu_data()[0] * this->blob_top_data2_->cpu_diff()[0] +
          this->blob_top_data3_->cpu_data()[0] * this->blob_top_data3_->cpu_diff()[0];
  this->bbox_pred->mutable_cpu_data()[offset] = x + esp;
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  loss2 = this->blob_top_data1_->cpu_data()[0] * this->blob_top_data1_->cpu_diff()[0] +
          this->blob_top_data2_->cpu_data()[0] * this->blob_top_data2_->cpu_diff()[0] +
          this->blob_top_data3_->cpu_data()[0] * this->blob_top_data3_->cpu_diff()[0];
  gradient2 = (loss2 - loss1) / (2 * esp);
  cout << "calculate gradient " << gradient2 << endl;
  cout << "backward gradient " << gradient1 << endl;
}

}  // namespace caffe
