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
  FILE *fin = fopen(text_file.c_str(), "r");
  if (!fin) {
    cout << "Can not open " << text_file << endl;
    exit(-1);
  }
  char path[300];
  vector<string> paths;
  while (fscanf(fin, "%s", path) > 0) {
    paths.push_back(path);
  }

  fclose(fin);
  return paths;
}

/*!
 * \breif detect face over data and output result to a text file
 */
void detect(vector<string> &paths, string out, int level) {
  // create detector
  int thread_n = omp_get_max_threads();
  vector<jfda::Detector*> detector_pool(thread_n);
  for (int i = 0; i < thread_n; i++) {
    detector_pool[i] = new jfda::Detector();
  }

  cout << "Start Detecting" << endl;
  int counter = 0;

  int n = paths.size();
  n = 200;
  vector<vector<jfda::FaceBBox> > result(n);
  //#pragma omp parallel for
  for (int i = 0; i < n; i++) {
    int thread_id = omp_get_thread_num();
    jfda::Detector* detector = detector_pool[thread_id];

    Mat img = imread(paths[i], CV_LOAD_IMAGE_COLOR);
    result[i] = detector->detect(img, level);

    for (int j = 0; j < result[i].size(); j++) {
      jfda::FaceBBox bbox = result[i][j];
      rectangle(img, Rect(bbox.x, bbox.y, bbox.w, bbox.h), Scalar(0, 0, 255));
    }
    imshow("img", img);
    waitKey(0);

    #pragma omp critical
    {
      counter++;
      if (counter % 100 == 0) {
        cout << "Processing " << counter << endl;
      }
    }
  }

  for (int i = 0; i < thread_n; i++) {
    delete detector_pool[i];
  }

  cout << "Write result to " << out << endl;
  FILE *fout = fopen(out.c_str(), "w");
  for (int i = 0; i < n; i++) {
    fprintf(fout, "%s\n", paths[i].c_str());
    vector<jfda::FaceBBox>& bboxes = result[i];
    fprintf(fout, "%d\n", bboxes.size());
    for (int j = 0; j < bboxes.size(); j++) {
      fprintf(fout, "%lf %lf %lf %lf\n", bboxes[j].x, bboxes[j].y, bboxes[j].w, bboxes[j].h);
    }
  }
  fclose(fout);
}


int main(int argc, char *argv[]) {
  cout << "Loading train data" << endl;
  vector<string> train_data = load_wider("../tmp/wider_train.txt");
  cout << "Loading val data" << endl;
  vector<string> val_data = load_wider("../tmp/wider_val.txt");
  detect(train_data, "../tmp/wider_p_train.txt", 1);
  detect(val_data, "../tmp/wider_p_val.txt", 1);
}
