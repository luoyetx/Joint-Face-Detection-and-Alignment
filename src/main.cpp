#include <vector>
#include <iostream>
#include <glog/logging.h>
#include <opencv2/highgui/highgui.hpp>
#include "detect.hpp"

using namespace cv;
using namespace std;
using namespace jfda;

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
