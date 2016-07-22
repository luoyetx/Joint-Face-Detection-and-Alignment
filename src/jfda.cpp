#include "jfda.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <caffe/caffe.hpp>

using namespace cv;
using namespace std;

namespace jfda {

vector<FaceBBox> nms(vector<FaceBBox>& bboxes, float overlap = 0.3);

Detector::Detector() {
  pnet = new caffe::Net<float>("../proto/p.prototxt", caffe::TEST);
  pnet->CopyTrainedLayersFromBinaryProto("../result/p.caffemodel");
  rnet = new caffe::Net<float>("../proto/r.prototxt", caffe::TEST);
  rnet->CopyTrainedLayersFromBinaryProto("../result/r.caffemodel");
  onet = new caffe::Net<float>("../proto/o.prototxt", caffe::TEST);
  onet->CopyTrainedLayersFromBinaryProto("../result/o.caffemodel");
}

Detector::~Detector() {
  delete pnet;
  delete rnet;
  delete onet;
}

static const float kPNetScoreTh = 0.5;
static const float kRNetScoreTh = 0.8;
static const float kONetScoreTh = 0.9;

vector<FaceBBox> Detector::detect(const Mat& img_, int level) {
  Mat img = img_.clone();
  //vector<Mat> bgr;
  //cv::split(img, bgr);
  int height = img.rows / 2;
  int width = img.cols / 2;
  cv::resize(img, img, cv::Size(width, height));
  float scale = 2.;
  float factor = 1.4;
  vector<FaceBBox> p_res;
  boost::shared_ptr<caffe::Blob<float> > input = pnet->blob_by_name("data");
  boost::shared_ptr<caffe::Blob<float> > face_prob = pnet->blob_by_name("face_prob");
  boost::shared_ptr<caffe::Blob<float> > bbox_offset = pnet->blob_by_name("face_bbox");
  TIMER_BEGIN
  int counter = 0;
  while (std::min(height, width) > 20) {
    counter++;
    input->Reshape(1, 3, height, width);
    float* input_data = input->mutable_cpu_data();
    for (int i = 0; i < height; i++) {
      for (int j = 0; j < width; j++) {
        input_data[input->offset(0, 0, i, j)] = static_cast<float>(img.at<cv::Vec3b>(i, j)[0]) / 128 - 1;
        input_data[input->offset(0, 1, i, j)] = static_cast<float>(img.at<cv::Vec3b>(i, j)[1]) / 128 - 1;
        input_data[input->offset(0, 2, i, j)] = static_cast<float>(img.at<cv::Vec3b>(i, j)[2]) / 128 - 1;
        //input_data[input->offset(0, 0, i, j)] = static_cast<float>(bgr[0].at<uchar>(i, j)) / 128 - 1;
        //input_data[input->offset(0, 1, i, j)] = static_cast<float>(bgr[1].at<uchar>(i, j)) / 128 - 1;
        //input_data[input->offset(0, 2, i, j)] = static_cast<float>(bgr[2].at<uchar>(i, j)) / 128 - 1;
      }
    }
    TIMER_BEGIN
    pnet->Forward();
    //cout << TIMER_NOW << endl;
    TIMER_END
    int h, w;
    h = face_prob->shape(2);
    w = face_prob->shape(3);
    float* face_prob_data = face_prob->mutable_cpu_data();
    float* bbox_offset_data = bbox_offset->mutable_cpu_data();
    for (int i = 0; i < h; i++) {
      for (int j = 0; j < w; j++) {
        float prob = face_prob_data[face_prob->offset(0, 1, i, j)];
        if (prob > kPNetScoreTh) {
          FaceBBox bbox;
          bbox.x = 2 * j*scale;
          bbox.y = 2 * i*scale;
          bbox.w = bbox.h = 12 * scale;

          bbox.x = bbox.x + bbox.w * bbox_offset_data[bbox_offset->offset(0, 0, i, j)];
          bbox.y = bbox.y + bbox.h * bbox_offset_data[bbox_offset->offset(0, 1, i, j)];
          bbox.w = bbox.w * exp(bbox_offset_data[bbox_offset->offset(0, 2, i, j)]);
          bbox.h = bbox.h * exp(bbox_offset_data[bbox_offset->offset(0, 2, i, j)]);
          bbox.score = prob;
          p_res.push_back(bbox);
        }
      }
    }

    scale *= factor;
    height = height / factor;
    width = width / factor;
    cv::resize(img, img, cv::Size(width, height));
    //cv::resize(bgr[0], bgr[0], cv::Size(width, height));
    //cv::resize(bgr[1], bgr[1], cv::Size(width, height));
    //cv::resize(bgr[2], bgr[2], cv::Size(width, height));
  }
  //cout << counter << endl;
  //cout << TIMER_NOW << endl;
  TIMER_END
  p_res = nms(p_res);

  if (level <= 1 || p_res.size() == 0) {
    return p_res;
  }

  // rnet
  int n = p_res.size();
  input = rnet->blob_by_name("data");
  face_prob = rnet->blob_by_name("face_prob");
  bbox_offset = rnet->blob_by_name("face_bbox");
  input->Reshape(n, 3, 24, 24);
  for (int k = 0; k < n; k++) {
    float* input_data = input->mutable_cpu_data();
    Mat patch;
    float x = p_res[k].x;
    float y = p_res[k].y;
    float w = p_res[k].w;
    float h = p_res[k].h;
    if (x < 0) x = 0;
    if (y < 0) y = 0;
    if (x + w > img_.cols) w = img_.cols - x;
    if (y + h > img_.rows) h = img_.rows - y;
    cv::resize(img_(cv::Rect(x, y, w, h)), patch, cv::Size(24, 24));
    for (int i = 0; i < 24; i++) {
      for (int j = 0; j < 24; j++) {
        input_data[input->offset(k, 0, i, j)] = static_cast<float>(patch.at<cv::Vec3b>(i, j)[0]) / 128 - 1;
        input_data[input->offset(k, 1, i, j)] = static_cast<float>(patch.at<cv::Vec3b>(i, j)[1]) / 128 - 1;
        input_data[input->offset(k, 2, i, j)] = static_cast<float>(patch.at<cv::Vec3b>(i, j)[2]) / 128 - 1;
      }
    }
  }
  TIMER_BEGIN
  rnet->Forward();
  //cout << "rnet forward time " << TIMER_NOW << endl;
  TIMER_END

  float* face_prob_data = face_prob->mutable_cpu_data();
  float* bbox_offset_data = bbox_offset->mutable_cpu_data();
  vector<FaceBBox> r_res;
  for (int i = 0; i < n; i++) {
    float prob = face_prob_data[face_prob->offset(i, 1, 0, 0)];
    if (prob > kRNetScoreTh) {
      float dx, dy, ds;
      dx = bbox_offset_data[bbox_offset->offset(i, 0, 0, 0)];
      dy = bbox_offset_data[bbox_offset->offset(i, 1, 0, 0)];
      ds = bbox_offset_data[bbox_offset->offset(i, 2, 0, 0)];
      p_res[i].x = p_res[i].x + p_res[i].w * dx;
      p_res[i].y = p_res[i].y + p_res[i].h * dy;
      p_res[i].w = p_res[i].w * exp(ds);
      p_res[i].h = p_res[i].h * exp(ds);
      p_res[i].score = prob;
      r_res.push_back(p_res[i]);
    }
  }

  r_res = nms(r_res);

  if (level <= 2 || r_res.size() == 0) {
    return r_res;
  }

  // onet
  n = r_res.size();
  input = onet->blob_by_name("data");
  face_prob = onet->blob_by_name("face_prob");
  bbox_offset = onet->blob_by_name("face_bbox");
  input->Reshape(n, 3, 48, 48);
  for (int k = 0; k < n; k++) {
    float* input_data = input->mutable_cpu_data();
    Mat patch;
    float x = r_res[k].x;
    float y = r_res[k].y;
    float w = r_res[k].w;
    float h = r_res[k].h;
    if (x < 0) x = 0;
    if (y < 0) y = 0;
    if (x + w > img_.cols) w = img_.cols - x;
    if (y + h > img_.rows) h = img_.rows - y;
    cv::resize(img_(cv::Rect(x, y, w, h)), patch, cv::Size(48, 48));
    for (int i = 0; i < 48; i++) {
      for (int j = 0; j < 48; j++) {
        input_data[input->offset(k, 0, i, j)] = static_cast<float>(patch.at<cv::Vec3b>(i, j)[0]) / 128 - 1;
        input_data[input->offset(k, 1, i, j)] = static_cast<float>(patch.at<cv::Vec3b>(i, j)[1]) / 128 - 1;
        input_data[input->offset(k, 2, i, j)] = static_cast<float>(patch.at<cv::Vec3b>(i, j)[2]) / 128 - 1;
      }
    }
  }
  TIMER_BEGIN
    onet->Forward();
    //cout << "onet forward time " << TIMER_NOW << endl;
  TIMER_END

  face_prob_data = face_prob->mutable_cpu_data();
  bbox_offset_data = bbox_offset->mutable_cpu_data();
  vector<FaceBBox> o_res;
  for (int i = 0; i < n; i++) {
    float prob = face_prob_data[face_prob->offset(i, 1, 0, 0)];
    if (prob > kONetScoreTh) {
      float dx, dy, ds;
      dx = bbox_offset_data[bbox_offset->offset(i, 0, 0, 0)];
      dy = bbox_offset_data[bbox_offset->offset(i, 1, 0, 0)];
      ds = bbox_offset_data[bbox_offset->offset(i, 2, 0, 0)];
      r_res[i].x = r_res[i].x + r_res[i].w * dx;
      r_res[i].y = r_res[i].y + r_res[i].h * dy;
      r_res[i].w = r_res[i].w * exp(ds);
      r_res[i].h = r_res[i].h * exp(ds);
      r_res[i].score = prob;
      o_res.push_back(r_res[i]);
    }
  }

  o_res = nms(o_res);

  return o_res;
}

vector<FaceBBox> nms(vector<FaceBBox>& bboxes, float overlap) {
  int n = bboxes.size();
  vector<FaceBBox> merged;
  merged.reserve(n);

  for (int i = 0; i < n; i++) {
    for (int j = i + 1; j < n; j++) {
      if (bboxes[i].score < bboxes[j].score) {
        std::swap(bboxes[i], bboxes[j]);
      }
    }
  }

  vector<float> areas(n);
  vector<bool> flag(n, true);
  for (int i = 0; i < n; i++) {
    areas[i] = bboxes[i].w*bboxes[i].h;
  }

  for (int i = 0; i < n; i++) {
    if (flag[i]) {
      merged.push_back(bboxes[i]);
      for (int j = i + 1; j < n; j++) {
        if (flag[j]) {
          float x1 = max(bboxes[i].x, bboxes[j].x);
          float y1 = max(bboxes[i].y, bboxes[j].y);
          float x2 = min(bboxes[i].x + bboxes[i].w, bboxes[j].x + bboxes[j].w);
          float y2 = min(bboxes[i].y + bboxes[i].h, bboxes[j].y + bboxes[j].h);
          float w = max(float(0), x2 - x1);
          float h = max(float(0), y2 - y1);
          float ov = (w*h) / (areas[i] + areas[j] - w*h);
          if (ov > overlap) {
            flag[j] = false;
          }
        }
      }
    }
  }
  return merged;
}

}  // namespace jfda
