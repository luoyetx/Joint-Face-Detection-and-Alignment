#include "caffe/layers/mask_identity_layer.hpp"

namespace caffe {

template<typename Dtype>
void MaskIdentityLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  top[0]->ReshapeLike(*bottom[0]);
}

template<typename Dtype>
void MaskIdentityLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const int n = bottom[0]->shape(0);
  const int m = bottom[0]->shape(1);
  const Dtype* bottom_ref = bottom[0]->cpu_data();
  const Dtype* bottom_data = bottom[1]->cpu_data();
  const Dtype* mask_data = bottom[2]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  for (int i = 0; i < n; i++) {
    const int offset = bottom[0]->offset(i);
    const Dtype* x_data = bottom_data + offset;
    const Dtype* x_ref = bottom_ref + offset;
    const int mask = static_cast<int>(mask_data[i]);
    Dtype* y = top_data + offset;
    for (int j = 0; j < m; j++) {
      // if (mask == 1) y[j] = x_ref[j];
      // else y[j] = x_data[j];
      y[j] = mask*x_ref[j] + (1-mask)*x_data[j];
    }
  }
}

template<typename Dtype>
void MaskIdentityLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& top) {
  // Do nothing
}

#ifdef CPU_ONLY
STUB_GPU(MaskIdentityLayer)
#endif  // CPU_ONLY

INSTANTIATE_CLASS(MaskIdentityLayer);
REGISTER_LAYER_CLASS(MaskIdentity);

}  // namespace caffe
