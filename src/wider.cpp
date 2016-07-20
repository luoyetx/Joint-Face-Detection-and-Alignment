#include <omp.h>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "jfda.hpp"

using namespace cv;
using namespace std;

/*!
 * \breif read image path list from a text file
 */
vector<string> load_wider(string text_file) {
  ifstream ifs(text_file);
  if (!ifs.is_open()) {
    cout << "Can not open " << text_file << endl;
    exit(-1);
  }

  vector<string> path;
  string img_path;
  while (!ifs.eof()) {
    ifs >> img_path;
    path.push_back(img_path);
  }
  ifs.close();

  return path;
}

/*!
 * \breif detect face over data and output result to a text file
 */
void detect(vector<string> &data, string out) {
  // create detector
  int thread_n = omp_get_max_threads();
  vector<jfda::Detector> detector_pool(thread_n);

  cout << "Start Detecting" << endl;
  int counter = 0;

  int n = data.size();
  vector<vector<Rect> > result(n);
  //#pragma omp parallel for
  for (int i = 3; i < n; i++) {
    int thread_id = omp_get_thread_num();
    vector<Rect> &bboxes = result[i];
    jfda::Detector &detector = detector_pool[thread_id];

    Mat img = imread(data[i], CV_LOAD_IMAGE_COLOR);
    vector<jfda::FaceBBox> res = detector.detect(img, 1);
    for (int j = 0; j < res.size(); j++) {
      Rect bbox(res[j].x, res[j].y, res[j].w, res[j].h);
      bboxes.push_back(bbox);
    }

    for (int j = 0; j < bboxes.size(); j++) {
      rectangle(img, bboxes[j], Scalar(0, 0, 255));
    }
    imshow("img", img);
    waitKey(0);

    #pragma omp critical
    {
      counter++;
      if (counter % 500 == 0) {
        cout << "Processing " << counter << endl;
      }
    }
  }

  cout << "Write result to " << out << endl;
}


int main(int argc, char *argv[]) {
  cout << "Loading train data" << endl;
  vector<string> train_data = load_wider("../tmp/wider_train.txt");
  cout << "Loading val data" << endl;
  vector<string> val_data = load_wider("../tmp/wider_val.txt");
  detect(train_data, "../tmp/wider_r_train.txt");
  detect(val_data, "../tmp/wider_r_val.txt");
}
