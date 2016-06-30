#include "caffe/layers/mask_identity_layer.hpp"

namespace caffe {

template<typename Dtype>
__global__ void MaskIdentityForward(const int m, const int nthread,
    const Dtype* x_ref, const Dtype* x_data, const Dtype* mask_data, Dtype* y) {
  CUDA_KERNEL_LOOP(idx, nthread) {
    const int sample_idx = idx / m;
    const int mask = static_cast<int>(mask_data[sample_idx]);
    y[idx] = mask*x_ref[idx] + (1-mask)*x_data[idx];
  }
}

template<typename Dtype>
void MaskIdentityLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const int n = bottom[0]->shape(0);
  const int m = bottom[0]->shape(1);
  const Dtype* bottom_ref = bottom[0]->gpu_data();
  const Dtype* bottom_data = bottom[1]->gpu_data();
  const Dtype* mask_data = bottom[2]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int nthread = n*m;
  MaskIdentityForward<Dtype><<<CAFFE_GET_BLOCKS(nthread),
      CAFFE_CUDA_NUM_THREADS>>>(m, nthread, bottom_ref, bottom_data, mask_data, top_data);
}

template<typename Dtype>
void MaskIdentityLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& top) {
  // Do nothing
}

INSTANTIATE_LAYER_GPU_FUNCS(MaskIdentityLayer);

}  // namespace caffe
