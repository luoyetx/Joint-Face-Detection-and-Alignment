#include <vector>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "caffe/caffe.hpp"

#define TIMER_BEGIN { double __time__ = clock();
#define TIMER_NOW ((clock() - __time__) / CLOCKS_PER_SEC * 1000)
#define TIMER_END }

using namespace cv;
using namespace std;

const float kPNetScoreThreshold = 0.8;

struct FaceBBox {
  float x, y, w, h;
  float score;
  float landmark[10];
};

struct Detector {
  Detector() {
    string p = "../proto/p.prototxt";
    pnet = new caffe::Net<float>(p, caffe::TEST);
    pnet->CopyTrainedLayersFromBinaryProto("../result/p.caffemodel");
  }
  ~Detector() {
    delete pnet;
  }

  vector<FaceBBox> detect(const Mat& img_) {
    Mat img = img_.clone();
    int height = img.rows;
    int width = img.cols;
    float scale = 1.;
    float factor = 1.4;
    vector<FaceBBox> res;
    boost::shared_ptr<caffe::Blob<float> > input = pnet->blob_by_name("data");
    boost::shared_ptr<caffe::Blob<float> > face_prob = pnet->blob_by_name("face_prob");
    boost::shared_ptr<caffe::Blob<float> > bbox_offset = pnet->blob_by_name("face_bbox");
    TIMER_BEGIN
    int counter = 0;
    while (std::min(img.cols, img.rows) > 20) {
      counter++;
      input->Reshape(1, 3, height, width);
      float* input_data = input->mutable_cpu_data();
      for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
          input_data[input->offset(0, 0, i, j)] = static_cast<float>(img.at<cv::Vec3b>(i, j)[0]) / 128 - 1;
          input_data[input->offset(0, 1, i, j)] = static_cast<float>(img.at<cv::Vec3b>(i, j)[1]) / 128 - 1;
          input_data[input->offset(0, 2, i, j)] = static_cast<float>(img.at<cv::Vec3b>(i, j)[2]) / 128 - 1;
        }
      }
      pnet->Forward();
      int h, w;
      h = face_prob->shape(2);
      w = face_prob->shape(3);
      float* face_prob_data = face_prob->mutable_cpu_data();
      float* bbox_offset_data = bbox_offset->mutable_cpu_data();
      for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
          float prob = face_prob_data[face_prob->offset(0, 1, i, j)];
          if (prob > kPNetScoreThreshold) {
            FaceBBox bbox;
            bbox.x = 2 * j*scale;
            bbox.y = 2 * i*scale;
            bbox.w = bbox.h = 12 * scale;

            bbox.x = bbox.x + bbox.w * bbox_offset_data[bbox_offset->offset(0, 0, i, j)];
            bbox.y = bbox.y + bbox.h * bbox_offset_data[bbox_offset->offset(0, 1, i, j)];
            bbox.w = bbox.w * exp(bbox_offset_data[bbox_offset->offset(0, 2, i, j)]);
            bbox.h = bbox.h * exp(bbox_offset_data[bbox_offset->offset(0, 2, i, j)]);
            bbox.score = prob;
            res.push_back(bbox);
          }
        }
      }

      scale *= factor;
      height = height / factor;
      width = width / factor;
      cv::resize(img, img, cv::Size(width, height));
    }
    cout << counter << endl;
    cout << TIMER_NOW << endl;
    TIMER_END
    return nms(res);
  }

  vector<FaceBBox> nms(vector<FaceBBox>& bboxes, float overlap=0.3) {
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
          if (flag[i]) {
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

  caffe::Net<float> *pnet;
};

int main(int argc, char *argv[]) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_alsologtostderr = false;

  Detector detector;
  Mat img = cv::imread("../test.jpg", cv::IMREAD_COLOR);

  cout << "start" << endl;
  vector<FaceBBox> res;
  TIMER_BEGIN
  res = detector.detect(img);
  cout << TIMER_NOW << endl;
  TIMER_END
  cout << "end\t" << endl;
  cout << "detect " << res.size() << " faces" << endl;

  for (int i = 0; i < res.size(); i++) {
    FaceBBox& bbox = res[i];
    cv::rectangle(img, Rect(bbox.x, bbox.y, bbox.w, bbox.h), Scalar(0, 0, 255), 1);
  }

  cv::imshow("image", img);
  cv::waitKey(0);

  return 0;
}
