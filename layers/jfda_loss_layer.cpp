#include <cmath>
#include <cfloat>
#include "caffe/layers/jfda_loss_layer.hpp"
#include <iostream>

using std::cout;
using std::endl;

namespace caffe {

// JfdaLossLayer
// LayerInput
//  1. face classification score
//  2. face bbox regression pred
//  3. face landmark pred
//  4. face bbox regression target
//  5. face landmark target
//  6. face data type / label, 0 for negatives, 1 for positives,
//                             2 for part faces, 3 for landmark faces
// LayerOutput
//  1. face classification loss
//  2. face bbox regression loss
//  3. face landmark loss
//  4. face classification accuracy x 2
// Training data layout
//  negatives, positives, part faces, landmark faces
// Bottom
//  bottom[0]: face classification score
//  bottom[1]: face bbox pred
//  bottom[2]: face landmark pred
//  bottom[3]: face bbox target
//  bottom[4]: face landmark target
//  bottom[5]: face data type / label
// Top
//  top[0]: face classificatoin loss
//  top[1]: face bbox regression loss
//  top[2]: face landmark regression loos
//  top[3]: face classification negative accuracy
//  top[4]: face classification positive accuracy

template<typename Dtype>
void JfdaLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  drop_loss_rate_ = this->layer_param_.jfda_loss_param().drop_loss_rate();
  if (drop_loss_rate_ < 0.f) drop_loss_rate_ = 0.f;
  if (drop_loss_rate_ > 1.f) drop_loss_rate_ = 1.f;
}

template<typename Dtype>
void JfdaLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  mask_.ReshapeLike(*bottom[5]);
  prob_.ReshapeLike(*bottom[0]);
  bbox_diff_.ReshapeLike(*bottom[1]);
  landmark_diff_.ReshapeLike(*bottom[2]);
  vector<int> loss_shape(0);
  top[0]->Reshape(loss_shape);
  top[1]->Reshape(loss_shape);
  top[2]->Reshape(loss_shape);
  top[3]->Reshape(loss_shape);
  top[4]->Reshape(loss_shape);
}

template<typename Dtype>
void _QSort_(vector<Dtype>& loss, vector<int>& idx, int left, int right) {
  int i = left;
  int j = right;
  Dtype t = loss[(i + j) / 2];
  do {
    while (loss[i] > t) i++;
    while (loss[j] < t) j--;
    if (i <= j) {
      std::swap(loss[i], loss[j]);
      std::swap(idx[i], idx[j]);
      i++;
      j--;
    }
  } while (i <= j);
  if (left < j) _QSort_(loss, idx, left, j);
  if (i < right) _QSort_(loss, idx, i, right);
}

template<typename Dtype>
void JfdaLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  int n1, n2, n3, n4;
  n1 = n2 = n3 = n4 = 0;
  const Dtype* label_data = bottom[5]->cpu_data();
  const int batch_size = bottom[5]->num();
  for (int i = 0; i < batch_size; i++) {
    const int label = static_cast<int>(label_data[i]);
    if (label == 0) n1++;
    else if (label == 1) n2++;
    else if (label == 2) n3++;
    else n4++;
  }

  // face classification
  int n = n1 + n2;
  const Dtype* score_data = bottom[0]->cpu_data();
  Dtype* prob_data = prob_.mutable_cpu_data();
  vector<Dtype> loss_data(n);
  Dtype face_cls_loss = 0;
  Dtype pos_acc = 0;
  Dtype neg_acc = 0;
  for (int i = 0; i < n; i++) {
    const Dtype max_input = std::max(score_data[2 * i], score_data[2 * i + 1]);
    prob_data[2 * i] = std::exp(score_data[2 * i] - max_input);
    prob_data[2 * i + 1] = std::exp(score_data[2 * i + 1] - max_input);
    const Dtype sum = prob_data[2 * i] + prob_data[2 * i + 1];
    prob_data[2 * i] /= sum;
    prob_data[2 * i + 1] /= sum;
    const int label = static_cast<int>(label_data[i]);
    if (label == 0) {
      loss_data[i] = -std::log(std::max(prob_data[2 * i], Dtype(FLT_MIN)));
      if (prob_data[2 * i] > prob_data[2 * i + 1]) neg_acc += 1;
    }
    else {
      loss_data[i] = -std::log(std::max(prob_data[2 * i + 1], Dtype(FLT_MIN)));
      if (prob_data[2 * i + 1] > prob_data[2 * i]) pos_acc += 1;
    }
    face_cls_loss += loss_data[i];
  }
  face_cls_loss /= batch_size;
  top[0]->mutable_cpu_data()[0] = face_cls_loss;
  neg_acc /= n1;
  pos_acc /= n2;
  top[3]->mutable_cpu_data()[0] = neg_acc;
  top[4]->mutable_cpu_data()[0] = pos_acc;

  // bbox regression
  // n = n1 + n2 + n3;
  // Dtype bbox_reg_loss = 0;
  // for (int i = n1; i < n; i++) {
  //   const Dtype* bbox_pred_data = bottom[1]->cpu_data() + bottom[1]->offset(i);
  //   const Dtype* bbox_target_data = bottom[3]->cpu_data() + bottom[3]->offset(i);
  //   const int m = bottom[1]->channels();
  //   for (int j = 0; j < m; j++) {
  //     bbox_reg_loss += (bbox_pred_data[j] - bbox_target_data[j]) *
  //                      (bbox_pred_data[j] - bbox_target_data[j]);
  //   }
  // }
  // bbox_reg_loss /= 2 * batch_size;
  // top[1]->mutable_cpu_data()[0] = bbox_reg_loss;

  const Dtype* bbox_pred_data = bottom[1]->cpu_data() + bottom[1]->offset(n1);
  const Dtype* bbox_target_data = bottom[3]->cpu_data() + bottom[3]->offset(n1);
  Dtype* bbox_diff_data = bbox_diff_.mutable_cpu_data() + bbox_diff_.offset(n1);
  const int bbox_count = (n2 + n3) * bottom[1]->channels();
  caffe_sub(
      bbox_count,
      bbox_pred_data,
      bbox_target_data,
      bbox_diff_data);
  Dtype bbox_dot = caffe_cpu_dot(bbox_count, bbox_diff_data, bbox_diff_data);
  Dtype bbox_reg_loss = bbox_dot / batch_size / Dtype(2);
  top[1]->mutable_cpu_data()[0] = bbox_reg_loss;
  // cout << "bbox pred [" << bbox_pred_data[0] << ", "
  //                       << bbox_pred_data[1] << ", "
  //                       << bbox_pred_data[2] << ", "
  //                       << bbox_pred_data[3] << "]" << endl;

  // landmark regression
  // n = n1 + n2 + n3 + n4;
  // Dtype landmark_reg_loss = 0;
  // for (int i = n1 + n2 + n3; i < n; i++) {
  //   const Dtype* landmark_pred_data = bottom[2]->cpu_data() + bottom[2]->offset(i);
  //   const Dtype* landmark_target_data = bottom[4]->cpu_data() + bottom[4]->offset(i);
  //   const int m = bottom[2]->channels();
  //   for (int j = 0; j < m; j++) {
  //     landmark_reg_loss += (landmark_pred_data[j] - landmark_target_data[j]) *
  //                          (landmark_pred_data[j] - landmark_target_data[j]);
  //   }
  // }
  // landmark_reg_loss /= 2 * batch_size;
  // top[2]->mutable_cpu_data()[1] = landmark_reg_loss;

  const Dtype* landmark_pred_data = bottom[2]->cpu_data() + bottom[2]->offset(n1 + n2 + n3);
  const Dtype* landmark_target_data = bottom[4]->cpu_data() + bottom[4]->offset(n1 + n2 + n3);
  Dtype* landmark_diff_data = landmark_diff_.mutable_cpu_data() + landmark_diff_.offset(n1 + n2 + n3);
  const int landmark_count = n4 * bottom[2]->channels();
  caffe_sub(
      landmark_count,
      landmark_pred_data,
      landmark_target_data,
      landmark_diff_data);
  Dtype landmark_dot = caffe_cpu_dot(landmark_count, landmark_diff_data, landmark_diff_data);
  Dtype landmark_reg_loss = landmark_dot / batch_size / Dtype(2);
  top[2]->mutable_cpu_data()[0] = landmark_reg_loss;

  // set backward mask for face classification
  vector<int> idx(loss_data.size());
  for (int i = 0; i < idx.size(); i++) {
    idx[i] = i;
  }
  _QSort_(loss_data, idx, 0, loss_data.size() - 1);
  const Dtype th = static_cast<Dtype>(1.f - drop_loss_rate_);
  const int remained = static_cast<int>(loss_data.size() * th);
  Dtype* mask_data = mask_.mutable_cpu_data();
  for (int i = 0; i < remained; i++) {
    mask_data[idx[i]] = Dtype(1);
  }
  for (int i = remained; i < loss_data.size(); i++) {
    mask_data[idx[i]] = Dtype(0);
  }
}

template<typename Dtype>
void JfdaLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  int n1, n2, n3, n4;
  n1 = n2 = n3 = n4 = 0;
  const Dtype* label_data = bottom[5]->cpu_data();
  const int batch_size = bottom[5]->num();
  for (int i = 0; i < batch_size; i++) {
    const int label = static_cast<int>(label_data[i]);
    if (label == 0) n1++;
    else if (label == 1) n2++;
    else if (label == 2) n3++;
    else n4++;
  }

  // face classification
  int n = n1 + n2;
  const Dtype* prob_data = prob_.cpu_data();
  const Dtype* mask_data = mask_.cpu_data();
  Dtype* cls_diff = bottom[0]->mutable_cpu_diff();
  caffe_copy(2 * n, prob_data, cls_diff);
  for (int i = 0; i < n; i++) {
    const int label = static_cast<int>(label_data[i]);
    const Dtype weight = mask_data[i] * top[0]->cpu_diff()[0] / batch_size;
    if (label == 0) {
      cls_diff[2 * i] -= 1;
    }
    else {
      cls_diff[2 * i + 1] -= 1;
    }
    cls_diff[2 * i] *= weight;
    cls_diff[2 * i + 1] *= weight;
  }

  // // bbox regression
  // n = n1 + n2 + n3;
  // for (int i = n1; i < n; i++) {
  //   const Dtype* bbox_pred_data = bottom[1]->cpu_data() + bottom[1]->offset(i);
  //   const Dtype* bbox_target_data = bottom[3]->cpu_data() + bottom[3]->offset(i);
  //   Dtype* reg_diff = bottom[1]->mutable_cpu_diff() + bottom[1]->offset(i);
  //   const int m = bottom[1]->channels();
  //   const Dtype weight = top[1]->cpu_diff()[0] /  batch_size;
  //   for (int j = 0; j < m; j++) {
  //     reg_diff[j] = weight * (bbox_pred_data[j] - bbox_target_data[j]);
  //   }
  // }

  const Dtype* bbox_diff_data = bbox_diff_.cpu_data() + bbox_diff_.offset(n1);
  Dtype* bbox_reg_diff = bottom[1]->mutable_cpu_diff() + bottom[1]->offset(n1);
  const int bbox_count = (n2 + n3) * bottom[1]->channels();
  const Dtype bbox_alpha = top[1]->cpu_diff()[0] / batch_size;
  caffe_cpu_axpby(
      bbox_count,
      bbox_alpha,
      bbox_diff_data,
      Dtype(0),
      bbox_reg_diff);

  // // landmark regression
  // n =  n1 + n2 + n3 + n4;
  // for (int i = n1 + n2 + n3; i < n; i++) {
  //   const Dtype* landmark_pred_data = bottom[2]->cpu_data() + bottom[2]->offset(i);
  //   const Dtype* landmark_target_data = bottom[4]->cpu_data() + bottom[4]->offset(i);
  //   Dtype* reg_diff = bottom[2]->mutable_cpu_diff() + bottom[2]->offset(i);
  //   const int m = bottom[1]->channels();
  //   const Dtype weight = top[2]->cpu_diff()[0] / batch_size;
  //   for (int j = 0; j < m; j++) {
  //     reg_diff[j] = weight * (landmark_pred_data - landmark_target_data);
  //   }
  // }

  const Dtype* landmark_diff_data = landmark_diff_.cpu_data() + landmark_diff_.offset(n1 + n2 + n3);
  Dtype* landmark_reg_loss = bottom[2]->mutable_cpu_diff() + bottom[2]->offset(n1 + n2 + n3);
  const int landmark_count = n4 * bottom[2]->channels();
  const Dtype landmark_alpha = top[2]->cpu_diff()[0] / batch_size;
  caffe_cpu_axpby(
      landmark_count,
      landmark_alpha,
      landmark_diff_data,
      Dtype(0),
      landmark_reg_loss);
}

#ifdef CPU_ONLY
STUB_GPU(JfdaLossLayer)
#endif  // CPU_ONLY

INSTANTIATE_CLASS(JfdaLossLayer);
REGISTER_LAYER_CLASS(JfdaLoss);

}  // namespace caffe
