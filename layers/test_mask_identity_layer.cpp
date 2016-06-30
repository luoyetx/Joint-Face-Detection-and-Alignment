#include <cmath>
#include <vector>
#include <algorithm>
#include <boost/scoped_ptr.hpp>
#include "gtest/gtest.h"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/mask_identity_layer.hpp"
#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

using boost::scoped_ptr;

namespace caffe {

template<typename TypeParam>
class MaskIdentityLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  MaskIdentityLayerTest()
      : blob_bottom_ref_(new Blob<Dtype>(100, 4, 1, 1)),
        blob_bottom_data_(new Blob<Dtype>(100, 4, 1, 1)),
        blob_bottom_mask_(new Blob<Dtype>(100, 1, 1, 1)),
        blob_top_data_(new Blob<Dtype>(100, 4, 1, 1)) {
    Caffe::set_random_seed(0);
    FillerParameter filler_param;
    filler_param.set_std(10);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_ref_);
    filler.Fill(this->blob_bottom_data_);
    for (int i = 0; i < blob_bottom_mask_->count(); i++) {
      blob_bottom_mask_->mutable_cpu_data()[i] = caffe_rng_rand() % 2;
    }
    blob_bottom_vec_.push_back(blob_bottom_ref_);
    blob_bottom_vec_.push_back(blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_mask_);
    blob_top_vec_.push_back(blob_top_data_);
  }

  virtual ~MaskIdentityLayerTest() {
    delete blob_bottom_ref_;
    delete blob_bottom_data_;
    delete blob_bottom_mask_;
    delete blob_top_data_;
  }

  Blob<Dtype>* const blob_bottom_ref_;
  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_bottom_mask_;
  Blob<Dtype>* const blob_top_data_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};  // class MaskIdentityTest

TYPED_TEST_CASE(MaskIdentityLayerTest, TestDtypesAndDevices);

TYPED_TEST(MaskIdentityLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  scoped_ptr<MaskIdentityLayer<Dtype> > layer(
    new MaskIdentityLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  const Dtype* bottom_ref = this->blob_bottom_ref_->cpu_data();
  const Dtype* bottom_data = this->blob_bottom_data_->cpu_data();
  const Dtype* mask_data = this->blob_bottom_mask_->cpu_data();
  const Dtype* top_data = this->blob_top_data_->cpu_data();

  const int n = this->blob_bottom_ref_->shape(0);
  const int m = this->blob_bottom_ref_->shape(1);
  for (int i = 0; i < n; i++) {
    const int offset = this->blob_bottom_ref_->offset(i);
    const Dtype* x_ref = bottom_ref + offset;
    const Dtype* x_data = bottom_data + offset;
    const Dtype* y = top_data + offset;
    const int mask = static_cast<int>(mask_data[i]);
    for (int j = 0; j < m; j++) {
      if (mask == 1) {
        EXPECT_EQ(y[j], x_ref[j]);
      }
      else {
        EXPECT_EQ(y[j], x_data[j]);
      }
    }
  }
}

}  // namespace caffe
