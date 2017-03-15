#include <ctime>
#include <chrono>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <caffe/profiler.hpp>
#include "jfda.hpp"

using namespace cv;
using namespace std;
using namespace jfda;

/*! \brief Timer */
class Timer {
  using Clock = std::chrono::high_resolution_clock;
public:
  /*! \brief start or restart timer */
  inline void Tic() {
    start_ = Clock::now();
  }
  /*! \brief stop timer */
  inline void Toc() {
    end_ = Clock::now();
  }
  /*! \brief return time in ms */
  inline double Elasped() {
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_ - start_);
    return duration.count();
  }

private:
  Clock::time_point start_, end_;
};

int main(int argc, char* argv[]) {
  string proto_dir = "../../proto/";
  string model_dir = "../../model/";
  JfdaDetector detector(proto_dir + "p.prototxt", model_dir + "p.caffemodel",
                        proto_dir + "r.prototxt", model_dir + "r.caffemodel",
                        proto_dir + "o.prototxt", model_dir + "o.caffemodel",
                        proto_dir + "l.prototxt", model_dir + "l.caffemodel", 0);

  Mat img = cv::imread("../../img/test1.jpg", CV_LOAD_IMAGE_COLOR);

  caffe::Profiler *profiler = caffe::Profiler::Get();
  profiler->TurnON();
  Timer timer;
  timer.Tic();
  detector.SetMinSize(40);
  vector<FaceInfo> faces = detector.Detect(img);
  timer.Toc();
  profiler->TurnOFF();
  profiler->DumpProfile("profile.json");

  cout << "detect costs " << timer.Elasped() << "ms" << endl;

  for (int i = 0; i < faces.size(); i++) {
    FaceInfo& face = faces[i];
    cv::rectangle(img, face.bbox, Scalar(0, 0, 255), 2);
    for (int j = 0; j < 5; j++) {
      cv::circle(img, face.landmark5[j], 2, Scalar(0, 255, 0), -1);
    }
  }
  cv::imwrite("result.jpg", img);
  cv::imshow("result", img);
  cv::waitKey(0);

  return 0;
}
