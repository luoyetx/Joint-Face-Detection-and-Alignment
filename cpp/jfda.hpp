#pragma once

#include <vector>
#include <string>
#include <memory>
#include <opencv2/core/core.hpp>

namespace jfda {

struct FaceInfo {
  cv::Rect bbox;
  float score;
  std::vector<cv::Point2f> landmark5;
};

/*!
 * \brief multi-task cascade CNN detector for face detection and alignment
 * \note  this object is not thread safe. every thread can only be binded to one gpu.
 *        we can create multi objects on the same gpu in one thread, but can not
 *        create multi objects on different gpus in one thread.
 */
class JfdaDetector {
public:
  /*!
   * \brief constructor
   * \param pnet, pmodel  caffe prototxt and caffemodel for pnet
   * \param rnet, rmodel  caffe prototxt and caffemodel for rnet
   * \param onet, omodel  caffe prototxt and caffemodel for onet
   * \param lnet, lmodel  caffe prototxt and caffemodel for lnet
   * \param gpu_device    gpu id for this model to run, -1 for cpu
   */
  JfdaDetector(const std::string& pnet, const std::string& pmodel,
               const std::string& rnet, const std::string& rmodel,
               const std::string& onet, const std::string& omodel,
               const std::string& lnet, const std::string& lmodel,
               int gpu_device=-1);
  /*!
   * \brief set minimum size of face, default 24
   * \param min_size  minimum size of face
   */
  void SetMinSize(int min_size);
  /*!
   * \brief set maximum size of face, default -1
   * \param max_size  maximum size of face, -1 for unlimited size which be bounded by image size
   */
  void SetMaxSize(int max_size);
  /*!
   * \brief set 3 threshold for pnet, rnet and onet, default 0.6, 0.7, 0.8
   * \param th1, th2, th3   network output threshold
   */
  void SetStageThresholds(float th1, float th2, float th3);
  /*!
   * \brief set image pyraimd scale factor, default 0.7
   * \param factor  image pyraimd scale factor
   */
  void SetImageScaleFactor(float factor);
  /*!
   * \brief set max size of image's width or height, resize will be performed
   * \param max_image_size  max size of image's width or height
   */
  void SetMaxImageSize(int max_image_size);
  /*!
   * \brief set pnet single forward
   * \note  this method should only be applied under GPU mode and input image is not too large
   * \param single_forward  set single forward flag
   */
  void SetPNetSingleForward(bool single_forward=true);
  /*!
   * \brief detect face
   * \param img   color image to detect face
   * \return      vector of face result
   */
  std::vector<FaceInfo> Detect(const cv::Mat& img);

private:
  class Impl;
  // although it's declared with shared_ptr, you should not use it through multi-thread, multi-object
  std::shared_ptr<Impl> impl_;
};

}  // namespace jfda
