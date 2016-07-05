#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "caffe/caffe.hpp"

using namespace cv;
using namespace std;

struct Detector {
  Detector() {
    string p = "../proto/p.prototxt";
    pnet = new caffe::Net<float>(p, caffe::TEST);
  }
  ~Detector() {
    delete pnet;
  }

  caffe::Net<float> *pnet;
};

int main(int argc, char *argv[]) {
  Detector detector;
  Mat img = cv::imread("../test.jpg", cv::IMREAD_COLOR);

  return 0;
}
