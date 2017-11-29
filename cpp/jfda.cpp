#include <iostream>
#include <functional>
#include <caffe/net.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <caffe/profiler.hpp>
#include "jfda.hpp"

using std::vector;
using std::string;
using caffe::Blob;
using std::shared_ptr;

namespace jfda {

cv::Mat CropPatch(const cv::Mat& img, cv::Rect& bbox) {
  int height = img.rows;
  int width = img.cols;
  int x1 = bbox.x;
  int y1 = bbox.y;
  int x2 = bbox.x + bbox.width;
  int y2 = bbox.y + bbox.height;
  cv::Mat patch = cv::Mat::zeros(bbox.height, bbox.width, img.type());
  // something stupid, totally out of boundary
  if (x1 >= width || y1 >= height || x2 <= 0 || y2 <= 0) {
    return patch;
  }
  // partially out of boundary
  if (x1 < 0 || y1 < 0 || x2 > width || y2 > height) {
    int vx1 = (x1 < 0 ? 0 : x1);
    int vy1 = (y1 < 0 ? 0 : y1);
    int vx2 = (x2 > width ? width : x2);
    int vy2 = (y2 > height ? height : y2);
    int sx = (x1 < 0 ? -x1 : 0);
    int sy = (y1 < 0 ? -y1 : 0);
    int vw = vx2 - vx1;
    int vh = vy2 - vy1;
    cv::Rect roi_src(vx1, vy1, vw, vh);
    cv::Rect roi_dst(sx, sy, vw, vh);
    img(roi_src).copyTo(patch(roi_dst));
  }
  else {
    img(bbox).copyTo(patch);
  }
  return patch;
}

class JfdaDetector::Impl {
public:
  Impl()
    : pnet(NULL), rnet(NULL), onet(NULL), lnet(NULL) {
  }

  ~Impl() {
    if (pnet) delete pnet;
    if (rnet) delete rnet;
    if (onet) delete onet;
    if (lnet) delete lnet;
  }

  /*! \brief internal detection */
  vector<FaceInfo> Detect(const cv::Mat& img);

public:
  int min_size_ = 24;
  int max_size_ = -1;
  float factor_ = 0.709;
  float th1_ = 0.6;
  float th2_ = 0.7;
  float th3_ = 0.8;
  bool single_forward = false;
  /*! \brief max size of image's width or height */
  int max_img_size_ = 640;
  /*! \brief caffe cnn networks */
  caffe::Net *pnet, *rnet, *onet, *lnet;
};

JfdaDetector::JfdaDetector(const string& pnet, const string& pmodel,
                           const string& rnet, const string& rmodel,
                           const string& onet, const string& omodel,
                           const string& lnet, const string& lmodel,
                           int gpu_device) {
  if (gpu_device < 0) {
    caffe::SetMode(caffe::CPU, -1);
  }
  else if (caffe::GPUAvailable()) {
    caffe::SetMode(caffe::GPU, gpu_device);
  }
  else {
    LOG(WARNING) << "GPU not available, fall back to CPU";
    caffe::SetMode(caffe::CPU, -1);
  }
  impl_.reset(new Impl);
  impl_->pnet = new caffe::Net(pnet);
  impl_->pnet->CopyTrainedLayersFrom(pmodel);
  impl_->rnet = new caffe::Net(rnet);
  impl_->rnet->CopyTrainedLayersFrom(rmodel);
  impl_->onet = new caffe::Net(onet);
  impl_->onet->CopyTrainedLayersFrom(omodel);
  impl_->lnet = new caffe::Net(lnet);
  impl_->lnet->CopyTrainedLayersFrom(lmodel);
}

void JfdaDetector::SetMinSize(int min_size) {
  if (min_size > 0) {
    impl_->min_size_ = min_size;
  }
}

void JfdaDetector::SetMaxSize(int max_size) {
  if (max_size > 0) {
    impl_->max_size_ = max_size;
  }
}

void JfdaDetector::SetImageScaleFactor(float factor) {
  if (factor > 0. && factor < 1.) {
    impl_->factor_ = factor;
  }
}

void JfdaDetector::SetStageThresholds(float th1, float th2, float th3) {
  if (th1 > 0. && th1 < 1.) {
    impl_->th1_ = th1;
  }
  if (th2 > 0. && th2 < 1.) {
    impl_->th2_ = th2;
  }
  if (th3 > 0. && th3 < 1.) {
    impl_->th3_ = th3;
  }
}

void JfdaDetector::SetMaxImageSize(int max_image_size) {
  if (max_image_size > 128) {
    impl_->max_img_size_ = max_image_size;
  }
}

void JfdaDetector::SetPNetSingleForward(bool single_forward) {
  impl_->single_forward = single_forward;
}

vector<FaceInfo> JfdaDetector::Detect(const cv::Mat& img) {
  // we need color image
  CV_Assert(img.type() == CV_8UC3);
  int w = img.cols;
  int h = img.rows;
  int max_size = impl_->max_img_size_;
  if (std::max(w, h) <= max_size) {
    return impl_->Detect(img);
  }
  // resize origin image
  cv::Mat img_small;
  float scale = 1.;
  if (w > h) {
    scale = w / static_cast<float>(max_size);
    h = static_cast<int>(h / scale);
    w = max_size;
  }
  else {
    scale = h / static_cast<float>(max_size);
    w = static_cast<int>(w / scale);
    h = max_size;
  }
  cv::resize(img, img_small, cv::Size(w, h));
  // detect
  vector<FaceInfo> faces = impl_->Detect(img_small);
  // scale back to origin image
  const int n = faces.size();
  for (int i = 0; i < n; i++) {
    FaceInfo& face = faces[i];
    face.bbox.x = static_cast<int>(face.bbox.x*scale);
    face.bbox.y = static_cast<int>(face.bbox.y*scale);
    face.bbox.width = static_cast<int>(face.bbox.width*scale);
    face.bbox.height = static_cast<int>(face.bbox.height*scale);
    for (int j = 0; j < 5; j++) {
      face.landmark5[j].x *= scale;
      face.landmark5[j].y *= scale;
    }
  }
  return faces;
}

/*! \brief structure for internal faces */
struct FaceInfoInternal {
  /*! \brief top left, right bottom bbox, [x1, y1, x2, y2] */
  float bbox[4];
  /*! \brief face score */
  float score;
  /*! \brief bbox offset [dx1, dy1, dx2, dy2] */
  float offset[4];
  /*! \brief landmark, left eye,, right eye, nose, left mouth, right mouth [x1, y1, x2, y2, ...] */
  float landmark[10];
};

/*!
 * \brief non-maximum suppression for face regions
 * \param faces     faces
 * \param th        threshold for overlap
 * \param is_union  whether to use union strategy or min strategy
 * \return          result faces after nms
 */
static vector<FaceInfoInternal> Nms(vector<FaceInfoInternal>& faces, float th, bool is_union=true);
/*!
 * \brief do bbox regression, x_new = x_old + dx * l
 * \param faces   faces, write result in place
 */
static void BBoxRegression(vector<FaceInfoInternal>& faces);
/*!
 * \brief make face bbox a square
 * \param faces   faces, write result in place
 */
static void BBoxSquare(vector<FaceInfoInternal>& faces);
/*!
 * \brief locate landmark in original image, points are aligned to top left of the image
 * \param faces   faces, write result in place
 */
static void LocateLandmark(vector<FaceInfoInternal>& faces);

/*! \brief information of image pyramid to a single image*/
struct PyramidInfo {
  int x, y, w, h;
  float scale;

  PyramidInfo(int x, int y, int w, int h, float scale)
    : x(x), y(y), w(w), h(h), scale(scale) {
  }
};
/*!
 * \brief convert image pyramid to a single image
 * \param img             original image
 * \param scales          pyramid scales, scales.size() >= 2
 * \param result          a single image convert from pyramid
 * \param pyramid_info    pyramid info about position in the single image
 */
void ConvertImagePyramid(const cv::Mat& img, const std::vector<float>& scales, \
                         cv::Mat& result, std::vector<PyramidInfo>& pyramid_info);

/*!
 * \brief just for debug, rely on `opencv/highgui` module
 */
static void debug_faces(const cv::Mat& img, vector<FaceInfoInternal>& faces, const char *fn=nullptr) {
  cv::Mat tmp;
  img.copyTo(tmp);
  for (int i = 0; i < faces.size(); i++) {
    FaceInfoInternal& face = faces[i];
    cv::Rect bbox(face.bbox[0], face.bbox[1], face.bbox[2] - face.bbox[0], face.bbox[3] - face.bbox[1]);
    cv::rectangle(tmp, bbox, cv::Scalar(0, 0, 255), 2);
  }
  if (fn == nullptr) {
    cv::imshow("tmp", tmp);
    cv::waitKey();
  }
  else {
    cv::imwrite(fn, tmp);
  }
}

void BatchSplit(caffe::Net* net, int n, int num_per_batch,
                std::function<void(int)> preprocess,
                std::function<void(int)> postprocess) {
  int num = n / num_per_batch;
  if (n%num_per_batch != 0) num++;

  for (int i = 0; i < num; i++) {
    preprocess(i);
    net->Forward();
    postprocess(i);
  }
}

vector<FaceInfo> JfdaDetector::Impl::Detect(const cv::Mat& img) {
  float base = 12.f / min_size_;
  int height = img.rows;
  int width = img.cols;
  // get image pyramid scales
  float l = std::min(height, width);
  if (max_size_ > 0) {
    l = std::min(l, static_cast<float>(max_size_));
  }
  l *= base;
  vector<float> scales;
  while (l > 12.f) {
    scales.push_back(base);
    base *= factor_;
    l *= factor_;
  }
  vector<FaceInfoInternal> faces;
  caffe::Profiler *profiler = caffe::Profiler::Get();
  profiler->ScopeStart("stage1");
  // stage-1
  if (!single_forward || scales.size() <= 1) {
    for (int i = 0; i < scales.size(); i++) {
      float scale = scales[i];
      int w = static_cast<int>(ceil(scale*width));
      int h = static_cast<int>(ceil(scale*height));
      cv::Mat data;
      vector<cv::Mat> bgr;
      cv::resize(img, data, cv::Size(w, h));
      cv::split(data, bgr);
      bgr[0].convertTo(bgr[0], CV_32F, 1.f / 128.f, -1.f);
      bgr[1].convertTo(bgr[1], CV_32F, 1.f / 128.f, -1.f);
      bgr[2].convertTo(bgr[2], CV_32F, 1.f / 128.f, -1.f);
      shared_ptr<Blob> input = pnet->blob_by_name("data");
      input->Reshape(1, 3, h, w);
      const int bias = input->offset(0, 1, 0, 0);
      const int bytes = bias*sizeof(float);
      memcpy(input->mutable_cpu_data() + 0 * bias, bgr[0].data, bytes);
      memcpy(input->mutable_cpu_data() + 1 * bias, bgr[1].data, bytes);
      memcpy(input->mutable_cpu_data() + 2 * bias, bgr[2].data, bytes);
      pnet->Forward();
      shared_ptr<Blob> prob = pnet->blob_by_name("prob");
      shared_ptr<Blob> bbox_offset = pnet->blob_by_name("bbox_pred");
      const int hm_h = prob->shape(2);
      const int hm_w = prob->shape(3);
      vector<FaceInfoInternal> faces_this_scale;
      for (int y = 0; y < hm_h; y++) {
        for (int x = 0; x < hm_w; x++) {
          if (prob->data_at(0, 1, y, x) > th1_) {
            FaceInfoInternal face;
            face.bbox[0] = 2 * x;
            face.bbox[1] = 2 * y;
            face.bbox[2] = 2 * x + 12;
            face.bbox[3] = 2 * y + 12;
            face.bbox[0] /= scale;
            face.bbox[1] /= scale;
            face.bbox[2] /= scale;
            face.bbox[3] /= scale;
            face.score = prob->data_at(0, 1, y, x);
            for (int j = 0; j < 4; j++) {
              face.offset[j] = bbox_offset->data_at(0, j, y, x);
            }
            faces_this_scale.push_back(face);
          }
        }
      }
      faces_this_scale = Nms(faces_this_scale, 0.5, true);
      faces.insert(faces.end(), faces_this_scale.begin(), faces_this_scale.end());
    }
  }
  else {
    cv::Mat data;
    vector<PyramidInfo> pyramid_info;
    ConvertImagePyramid(img, scales, data, pyramid_info);
    data.convertTo(data, CV_32F, 1.f / 128.f, -1.f);
    vector<cv::Mat> bgr;
    cv::split(data, bgr);
    shared_ptr<Blob> input = pnet->blob_by_name("data");
    input->Reshape(1, 3, data.rows, data.cols);
    const int bias = input->offset(0, 1, 0, 0);
    const int bytes = bias*sizeof(float);
    memcpy(input->mutable_cpu_data() + 0 * bias, bgr[0].data, bytes);
    memcpy(input->mutable_cpu_data() + 1 * bias, bgr[1].data, bytes);
    memcpy(input->mutable_cpu_data() + 2 * bias, bgr[2].data, bytes);
    pnet->Forward();
    shared_ptr<Blob> prob = pnet->blob_by_name("prob");
    shared_ptr<Blob> bbox_offset = pnet->blob_by_name("bbox_pred");
    const int hm_h = prob->shape(2);
    const int hm_w = prob->shape(3);
    for (int y = 0; y < hm_h; y++) {
      for (int x = 0; x < hm_w; x++) {
        if (prob->data_at(0, 1, y, x) > th1_) {
          int x1 = 2 * x;
          int y1 = 2 * y;
          int x2 = 2 * x + 12;
          int y2 = 2 * y + 12;
          // choosed a scale
          for (auto& info : pyramid_info) {
            if ((x1 >= info.x) && (y1 >= info.y) &&
                (x2 <= info.x + info.w) && (y2 <= info.y + info.h)) {
              FaceInfoInternal face;
              face.bbox[0] = (x1 - info.x) / info.scale;
              face.bbox[1] = (y1 - info.y) / info.scale;
              face.bbox[2] = (x2 - info.x) / info.scale;
              face.bbox[3] = (y2 - info.y) / info.scale;
              for (int j = 0; j < 4; j++) {
                face.offset[j] = bbox_offset->data_at(0, j, y, x);
              }
              faces.push_back(face);
              break;
            }
          }
        }
      }
    }
    faces = Nms(faces, 0.5, true);
  }
  faces = Nms(faces, 0.7, true);
  BBoxRegression(faces);
  BBoxSquare(faces);
  profiler->ScopeEnd();
  // stage-2
  int n = faces.size();
  if (n == 0) {
    return vector<FaceInfo>();
  }
  profiler->ScopeStart("stage2");
  vector<FaceInfoInternal> faces_stage2;
  int num_per_batch = 512;
  BatchSplit(rnet, n, num_per_batch,
    [&](int idx) {
      int begin = idx * num_per_batch;
      int end = (idx + 1) * num_per_batch;
      end = std::min(end, n);
      int num_valid = end - begin;
      shared_ptr<Blob> input = rnet->blob_by_name("data");
      input->Reshape(num_valid, 3, 24, 24);
      float* input_data = input->mutable_cpu_data();
      for (int i = begin; i < end; i++) {
        FaceInfoInternal& face = faces[i];
        cv::Rect bbox(face.bbox[0], face.bbox[1], face.bbox[2] - face.bbox[0], face.bbox[3] - face.bbox[1]);
        cv::Mat patch = CropPatch(img, bbox);
        cv::resize(patch, patch, cv::Size(24, 24));
        vector<cv::Mat> bgr;
        cv::split(patch, bgr);
        bgr[0].convertTo(bgr[0], CV_32F, 1.f / 128.f, -1.f);
        bgr[1].convertTo(bgr[1], CV_32F, 1.f / 128.f, -1.f);
        bgr[2].convertTo(bgr[2], CV_32F, 1.f / 128.f, -1.f);
        const int bytes = input->offset(0, 1)*sizeof(float);
        memcpy(input_data + input->offset(i - begin, 0), bgr[0].data, bytes);
        memcpy(input_data + input->offset(i - begin, 1), bgr[1].data, bytes);
        memcpy(input_data + input->offset(i - begin, 2), bgr[2].data, bytes);
      }
    },
    [&](int idx) {
      int begin = idx * num_per_batch;
      int end = (idx + 1) * num_per_batch;
      end = std::min(end, n);
      int num_valid = end - begin;
      shared_ptr<Blob> prob = rnet->blob_by_name("prob");
      shared_ptr<Blob> bbox_offset = rnet->blob_by_name("bbox_pred");
      for (int i = begin; i < end; i++) {
        if (prob->data_at(i - begin, 1, 0, 0) > th2_) {
          FaceInfoInternal face = faces[i];
          face.score = prob->data_at(i - begin, 1, 0, 0);
          for (int j = 0; j < 4; j++) {
            face.offset[j] = bbox_offset->data_at(i - begin, j, 0, 0);
          }
          faces_stage2.push_back(face);
        }
      }
    });
  faces_stage2 = Nms(faces_stage2, 0.7);
  BBoxRegression(faces_stage2);
  BBoxSquare(faces_stage2);
  faces = faces_stage2;
  profiler->ScopeEnd();
  // stage-3
  n = faces.size();
  if (n == 0) {
    return vector<FaceInfo>();
  }
  profiler->ScopeStart("stage3");
  vector<FaceInfoInternal> faces_stage3;
  num_per_batch = 128;
  BatchSplit(onet, n, num_per_batch,
    [&](int idx) {
      int begin = idx * num_per_batch;
      int end = (idx + 1) * num_per_batch;
      end = std::min(end, n);
      int num_valid = end - begin;
      shared_ptr<Blob> input = onet->blob_by_name("data");
      input->Reshape(num_valid, 3, 48, 48);
      float* input_data = input->mutable_cpu_data();
      for (int i = begin; i < end; i++) {
        FaceInfoInternal& face = faces[i];
        cv::Rect bbox(face.bbox[0], face.bbox[1], face.bbox[2] - face.bbox[0], face.bbox[3] - face.bbox[1]);
        cv::Mat patch = CropPatch(img, bbox);
        cv::resize(patch, patch, cv::Size(48, 48));
        vector<cv::Mat> bgr;
        cv::split(patch, bgr);
        bgr[0].convertTo(bgr[0], CV_32F, 1.f / 128.f, -1.f);
        bgr[1].convertTo(bgr[1], CV_32F, 1.f / 128.f, -1.f);
        bgr[2].convertTo(bgr[2], CV_32F, 1.f / 128.f, -1.f);
        const int bytes = input->offset(0, 1)*sizeof(float);
        memcpy(input_data + input->offset(i - begin, 0), bgr[0].data, bytes);
        memcpy(input_data + input->offset(i - begin, 1), bgr[1].data, bytes);
        memcpy(input_data + input->offset(i - begin, 2), bgr[2].data, bytes);
      }
    },
    [&](int idx) {
      int begin = idx * num_per_batch;
      int end = (idx + 1) * num_per_batch;
      end = std::min(end, n);
      int num_valid = end - begin;
      shared_ptr<Blob> prob = onet->blob_by_name("prob");
      shared_ptr<Blob> bbox_offset = onet->blob_by_name("bbox_pred");
      shared_ptr<Blob> landmark = onet->blob_by_name("landmark_pred");
      for (int i = begin; i < end; i++) {
        if (prob->data_at(i - begin, 1, 0, 0) > th3_) {
          FaceInfoInternal face = faces[i];
          face.score = prob->data_at(i - begin, 1, 0, 0);
          for (int j = 0; j < 4; j++) {
            face.offset[j] = bbox_offset->data_at(i - begin, j, 0, 0);
          }
          for (int j = 0; j < 10; j++) {
            face.landmark[j] = landmark->data_at(i - begin, j, 0, 0);
          }
          faces_stage3.push_back(face);
        }
      }
    });
  LocateLandmark(faces_stage3);
  BBoxRegression(faces_stage3);
  faces_stage3 = Nms(faces_stage3, 0.7, false);
  faces = faces_stage3;
  profiler->ScopeEnd();
  // stage-4
  n = faces.size();
  if (n == 0) {
    return vector<FaceInfo>();
  }
  profiler->ScopeStart("stage4");
  num_per_batch = 128;
  BatchSplit(lnet, n, num_per_batch,
    [&](int idx) {
      int begin = idx * num_per_batch;
      int end = (idx + 1) * num_per_batch;
      end = std::min(end, n);
      int num_valid = end - begin;
      shared_ptr<Blob> input = lnet->blob_by_name("data");
      input->Reshape(num_valid, 15, 24, 24);
      float* input_data = input->mutable_cpu_data();
      for (int i = begin; i < end; i++) {
        FaceInfoInternal& face = faces[i];
        float* landmark = face.landmark;
        int l = static_cast<int>(std::max(face.bbox[2] - face.bbox[0], face.bbox[3] - face.bbox[1]) * 0.25f);
        // every landmark patch
        for (int j = 0; j < 5; j++) {
          float x = landmark[2 * j];
          float y = landmark[2 * j + 1];
          cv::Rect patch_bbox(x - l / 2, y - l / 2, l, l);
          cv::Mat patch = CropPatch(img, patch_bbox);
          cv::resize(patch, patch, cv::Size(24, 24));
          vector<cv::Mat> bgr;
          cv::split(patch, bgr);
          bgr[0].convertTo(bgr[0], CV_32F, 1.f / 128.f, -1.f);
          bgr[1].convertTo(bgr[1], CV_32F, 1.f / 128.f, -1.f);
          bgr[2].convertTo(bgr[2], CV_32F, 1.f / 128.f, -1.f);
          const int bytes = input->offset(0, 1)*sizeof(float);
          memcpy(input_data + input->offset(i - begin, 3 * j + 0), bgr[0].data, bytes);
          memcpy(input_data + input->offset(i - begin, 3 * j + 1), bgr[1].data, bytes);
          memcpy(input_data + input->offset(i - begin, 3 * j + 2), bgr[2].data, bytes);
        }
      }
    },
    [&](int idx) {
      int begin = idx * num_per_batch;
      int end = (idx + 1) * num_per_batch;
      end = std::min(end, n);
      int num_valid = end - begin;
      shared_ptr<Blob> landmark_offset = lnet->blob_by_name("landmark_offset");
      for (int i = begin; i < end; i++) {
        FaceInfoInternal& face = faces[i];
        int l = static_cast<int>(std::max(face.bbox[2] - face.bbox[0], face.bbox[3] - face.bbox[1]) * 0.25f);
        for (int j = 0; j < 10; j++) {
          face.landmark[j] += landmark_offset->data_at(i - begin, j, 0, 0) * l;
        }
      }
    });
  // output
  n = faces.size();
  vector<FaceInfo> result(n);
  for (int i = 0; i < n; i++) {
    FaceInfoInternal& face = faces[i];
    FaceInfo& res = result[i];
    res.score = face.score;
    res.bbox.x = face.bbox[0];
    res.bbox.y = face.bbox[1];
    res.bbox.width = face.bbox[2] - face.bbox[0];
    res.bbox.height = face.bbox[3] - face.bbox[1];
    res.landmark5.clear();
    for (int j = 0; j < 10; j += 2) {
      res.landmark5.push_back(cv::Point2f(face.landmark[j], face.landmark[j + 1]));
    }
    // refine the bbox
    float x_min = img.cols;
    float x_max = -1;
    float y_min = img.rows;
    float y_max = -1;
    for (int j = 0; j < 5; j++) {
      x_min = std::min(x_min, res.landmark5[j].x);
      x_max = std::max(x_max, res.landmark5[j].x);
      y_min = std::min(y_min, res.landmark5[j].y);
      y_max = std::max(y_max, res.landmark5[j].y);
    }
    float w = x_max - x_min;
    float h = y_max - y_min;
    float r = 0.5;
    float s = std::max(w, h);
    int x = static_cast<int>(x_min - r*s);
    int y = static_cast<int>(y_min - r*s);
    s *= 1.f + 2.f * r;
    res.bbox.x = x;
    res.bbox.y = y;
    res.bbox.width = res.bbox.height = static_cast<int>(s);
  }
  profiler->ScopeEnd();
  return result;
}

vector<FaceInfoInternal> Nms(vector<FaceInfoInternal>& faces, float th, bool is_union) {
  typedef std::multimap<float, int> ScoreMapper;
  ScoreMapper sm;
  const int n = faces.size();
  vector<float> areas(n);
  for (int i = 0; i < n; i++) {
    FaceInfoInternal& face = faces[i];
    sm.insert(ScoreMapper::value_type(face.score, i));
    areas[i] = (face.bbox[3] - face.bbox[1])*(face.bbox[2] - face.bbox[0]);
  }
  vector<int> picked;
  while (!sm.empty()) {
    int last = sm.rbegin()->second;
    picked.push_back(last);
    FaceInfoInternal& last_face = faces[last];
    for (ScoreMapper::iterator it = sm.begin(); it != sm.end();) {
      int idx = it->second;
      FaceInfoInternal& face = faces[idx];
      float x1 = std::max(face.bbox[0], last_face.bbox[0]);
      float y1 = std::max(face.bbox[1], last_face.bbox[1]);
      float x2 = std::min(face.bbox[2], last_face.bbox[2]);
      float y2 = std::min(face.bbox[3], last_face.bbox[3]);
      float w = std::max(0.f, x2 - x1);
      float h = std::max(0.f, y2 - y1);
      float ov = (is_union ? (w*h) / (areas[idx] + areas[last] - w*h)
                           : (w*h) / std::min(areas[idx], areas[last]));
      if (ov > th) {
        ScoreMapper::iterator it_ = it;
        it_++;
        sm.erase(it);
        it = it_;
      }
      else {
        it++;
      }
    }
  }
  const int m = picked.size();
  vector<FaceInfoInternal> result(m);
  for (int i = 0; i < m; i++) {
    result[i] = faces[picked[i]];
  }
  return result;
}

void BBoxRegression(vector<FaceInfoInternal>& faces) {
  const int n = faces.size();
  for (int i = 0; i < n; i++) {
    FaceInfoInternal& face = faces[i];
    float w = face.bbox[2] - face.bbox[0];
    float h = face.bbox[3] - face.bbox[1];
    face.bbox[0] += face.bbox[5] * w;
    face.bbox[1] += face.bbox[6] * h;
    face.bbox[2] += face.bbox[7] * w;
    face.bbox[3] += face.bbox[8] * h;
  }
}

void BBoxSquare(vector<FaceInfoInternal>& faces) {
  const int n = faces.size();
  for (int i = 0; i < n; i++) {
    FaceInfoInternal& face = faces[i];
    float x_center = (face.bbox[0] + face.bbox[2]) / 2.f;
    float y_center = (face.bbox[1] + face.bbox[3]) / 2.f;
    float w = face.bbox[2] - face.bbox[0];
    float h = face.bbox[3] - face.bbox[1];
    float l = std::max(w, h);
    face.bbox[0] = x_center - l / 2.f;
    face.bbox[1] = y_center - l / 2.f;
    face.bbox[2] = x_center + l / 2.f;
    face.bbox[3] = y_center + l / 2.f;
  }
}

void LocateLandmark(vector<FaceInfoInternal>& faces) {
  const int n = faces.size();
  for (int i = 0; i < n; i++) {
    FaceInfoInternal& face = faces[i];
    float w = face.bbox[2] - face.bbox[0];
    float h = face.bbox[3] - face.bbox[1];
    for (int j = 0; j < 10; j += 2) {
      face.landmark[j] = face.bbox[0] + face.landmark[j] * w;
      face.landmark[j + 1] = face.bbox[1] + face.landmark[j + 1] * h;
    }
  }
}

void ConvertImagePyramid(const cv::Mat& img, const std::vector<float>& scales, \
                         cv::Mat& result, std::vector<PyramidInfo>& pyramid_info) {
  //CHECKGE(scales.size(), 2);
  std::vector<cv::Mat> pyramids;
  const int n = scales.size();
  const int height = img.rows;
  const int width = img.cols;
  const int interval = 2;
  pyramids.resize(n);
  pyramid_info.reserve(n);
  for (int i = 0; i < n; i++) {
    const float scale = scales[i];
    int h = static_cast<int>(std::ceil(height*scale));
    int w = static_cast<int>(std::ceil(width*scale));
    cv::resize(img, pyramids[i], cv::Size(w, h));
    pyramid_info.push_back(PyramidInfo(0, 0, w, h, scale));
  }
  const int input_h = pyramids[0].rows;
  const int input_w = pyramids[0].cols;
  int output_h, output_w;
  // available position (x, y)
  std::vector<std::pair<int, int> > available;
  available.push_back(std::make_pair(0, 0));
  if (input_h < input_w) {
    output_h = input_h + interval + pyramids[1].rows;
    output_w = 0;
    for (auto& info : pyramid_info) {
      int choosed = -1;
      int min_used_width = 3 * width;
      for (int i = 0; i < available.size(); i++) {
        auto& pos = available[i];
        const int x = pos.first;
        const int y = pos.second;
        if ((y + info.h <= output_h) && (x + info.w < min_used_width)) {
          min_used_width = x + info.w;
          info.x = x;
          info.y = y;
          choosed = i;
        }
      }
      //CHECKEQ(choosed, -1) << "No suitable position for this pyramid";
      // extend available positions
      const int x = available[choosed].first;
      const int y = available[choosed].second;
      const int w = info.w;
      const int h = info.h;
      available[choosed].first = x + w + interval;
      available[choosed].second = y;
      available.push_back(std::make_pair(x, y + h + interval));
      output_w = std::max(output_w, min_used_width);
    }
  }
  else {
    // same thing as above
    output_w = input_w + interval + pyramids[1].cols;
    output_h = 0;
    for (auto& info : pyramid_info) {
      int choosed = -1;
      int min_used_height = 2 * height;
      for (int i = 0; i < available.size(); i++) {
        auto& pos = available[i];
        const int x = pos.first;
        const int y = pos.second;
        if ((x + info.w <= output_w) && (y + info.h < min_used_height)) {
          min_used_height = y + info.h;
          info.x = x;
          info.y = y;
          choosed = i;
        }
      }
      //CHECKEQ(choosed, -1) << "No suitable position for this pyramid";
      // extend available positions
      const int x = available[choosed].first;
      const int y = available[choosed].second;
      const int w = info.w;
      const int h = info.h;
      available[choosed].first = x + w + interval;
      available[choosed].second = y;
      available.push_back(std::make_pair(x, y + h + interval));
      output_h = std::max(output_h, min_used_height);
    }
  }
  // convert to a single image
  result = cv::Mat(output_h, output_w, img.type());
  result.setTo(0);
  for (int i = 0; i < n; i++) {
    auto& info = pyramid_info[i];
    cv::Rect roi(info.x, info.y, info.w, info.h);
    pyramids[i].copyTo(result(roi));
  }
}

}  // namespace jfda
