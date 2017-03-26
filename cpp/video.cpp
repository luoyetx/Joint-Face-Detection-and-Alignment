#include <iostream>
#include "./jfda.hpp"
#include "caffe/caffe.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;
using namespace jfda;

int main(int argc, char* argv[]) {
  VideoCapture cap("./test.mp4");
  if (!cap.isOpened()) {
    cout << "Can\'t open test.mp4" << endl;
    return 0;
  }
  double rate = cap.get(CV_CAP_PROP_FPS);
  double width = cap.get(CV_CAP_PROP_FRAME_WIDTH);
  double height = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
  int fourcc = CV_FOURCC('X', 'V', 'I', 'D');
  VideoWriter writer("./result.avi", fourcc, rate, Size(width, height), true);
  string proto_dir = "../../proto/";
  string model_dir = "../../model/";
  JfdaDetector detector(proto_dir + "p.prototxt", model_dir + "p.caffemodel",
                        proto_dir + "r.prototxt", model_dir + "r.caffemodel",
                        proto_dir + "o.prototxt", model_dir + "o.caffemodel",
                        proto_dir + "l.prototxt", model_dir + "l.caffemodel", 0);
  detector.SetMinSize(24);
  Mat frame;
  caffe::Profiler* profiler = caffe::Profiler::Get();
  while (cap.read(frame)) {
    int64_t t1 = profiler->Now();
    vector<FaceInfo> faces = detector.Detect(frame);
    int64_t t2 = profiler->Now();
    double fps = 1000000 / (t2 - t1);
    for (int i = 0; i < faces.size(); i++) {
      FaceInfo& face = faces[i];
      cv::rectangle(frame, face.bbox, Scalar(0, 0, 255), 2);
      for (int j = 0; j < 5; j++) {
        cv::circle(frame, face.landmark5[j], 2, Scalar(0, 255, 0), -1);
      }
    }
    char buff[30];
    sprintf(buff, "%.2f FPS", fps);
    cv::putText(frame, buff, Point2f(0, 10), FONT_HERSHEY_PLAIN, 1, Scalar(0, 255, 0));
    writer.write(frame);
    cv::imshow("frame", frame);
    if (waitKey(1) >= 0) {
      break;
    }
  }
  return 0;
}
