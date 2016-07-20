#include <vector>
#include <opencv2/core/core.hpp>

#define TIMER_BEGIN { double __time__ = clock();
#define TIMER_NOW     ((clock() - __time__) / CLOCKS_PER_SEC * 1000)
#define TIMER_END   }

namespace caffe {
template<typename Dtype>
class Net;
}  // namespace caffe

namespace jfda {

struct FaceBBox {
  float x, y, w, h;
  float score;
  float landmark[10];
};

class Detector {
 public:
  Detector();
  ~Detector();

  std::vector<FaceBBox> detect(const cv::Mat& img, int level = 1);

 private:
  caffe::Net<float>* pnet;
  caffe::Net<float>* rnet;
  caffe::Net<float>* onet;
};

}  // namespace jfda
