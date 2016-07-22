#include <cstdio>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "jfda.hpp"

using namespace cv;
using namespace std;
using namespace jfda;

int main(int argc, char *argv[]) {
  const string fddb_dir = "../data/fddb";
  const string prefix = fddb_dir + "/images/";
  const string result_prefix = fddb_dir + "/result/out";

  jfda::Detector detector;

  for (int i = 1; i <= 10; i++) {
    char fddb[300];
    char fddb_out[300];
    char fddb_answer[300];

    printf("Testing FDDB-fold-%02d.txt\n", i);
    sprintf(fddb, "%s/FDDB-folds/FDDB-fold-%02d.txt", fddb_dir.c_str(), i);
    sprintf(fddb_out, "%s/result/fold-%02d-out.txt", fddb_dir.c_str(), i);
    sprintf(fddb_answer, "%s/FDDB-folds/FDDB-fold-%02d-ellipseList.txt", fddb_dir.c_str(), i);

    FILE* fin = fopen(fddb, "r");
    FILE* fanswer = fopen(fddb_answer, "r");
    if (!fin || !fanswer) {
      printf("Can not open file\n");
      exit(-1);
    }

#ifdef WIN32
    FILE* fout = fopen(fddb_out, "wb"); // replace \r\n on Windows platform
#else
    FILE* fout = fopen(fddb_out, "w");
#endif // WIN32

    char buff[300];
    char _buff[30];
    char path[300];

    int counter = 0;
    while (fscanf(fin, "%s", path) > 0) {
      string full_path = prefix + string(path) + string(".jpg");
      Mat img = imread(full_path, CV_LOAD_IMAGE_COLOR);
      if (!img.data) {
        printf("Can not open %s, Skip it", full_path.c_str());
        continue;
      }

      double fps = 0.;
      vector<jfda::FaceBBox> result = detector.detect(img, 3);
      const int n = result.size();

      fprintf(fout, "%s\n%d\n", path, n);

      for (int j = 0; j < n; j++) {
        const FaceBBox& r = result[j];
        fprintf(fout, "%d %d %d %d %lf\n", r.x, r.y, r.w, r.h, r.score);
      }

      counter++;
      sprintf(buff, "%s/%s.jpg", result_prefix.c_str(), path);

      // get answer
      int face_n = 0;
      fscanf(fanswer, "%s", path);
      fscanf(fanswer, "%d", &face_n);
      for (int k = 0; k < face_n; k++) {
        double major_axis_radius, minor_axis_radius, angle, center_x, center_y, score;
        fscanf(fanswer, "%lf %lf %lf %lf %lf %lf", &major_axis_radius, &minor_axis_radius, \
                                                    &angle, &center_x, &center_y, &score);
        // draw answer
        angle = angle / 3.1415926*180.;
        cv::ellipse(img, Point2d(center_x, center_y), Size(major_axis_radius, minor_axis_radius), \
                    angle, 0., 360., Scalar(255, 0, 0), 2);
      }

      // draw result
      for (int j = 0; j < n; j++) {
        const FaceBBox& r = result[j];
        double score = r.score;
        cv::rectangle(img, Rect(r.x, r.y, r.w, r.h), Scalar(0, 0, 255), 3);
        // draw score
        sprintf(_buff, "%.4lf", score);
        cv::putText(img, _buff, cv::Point(r.x, r.y), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 255, 0), 2);
        //// draw shape
        //for (int k = 0; k < 5; k++) {
        //  cv::circle(img, Point(r.landmark[2 * k], r.landmark[2 * k + 1]), 3, Scalar(0, 255, 0), -1);
        //}
      }

      string fname(path);
      std::replace(fname.begin(), fname.end(), '/', '_');
      fname = result_prefix + "/" + fname + ".jpg";
      cv::imwrite(fname, img);
    }
  }

  return 0;
}
