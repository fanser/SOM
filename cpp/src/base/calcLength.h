#pragma once
#include <vector>

#include <opencv2/opencv.hpp>

namespace SOM {
class CalcLength {
  CalcLength() {}
  virtual ~CalcLength() {}

  virtual std::vector<std::vector<float> > GetLength(const cv::Mat mat1, cosnt cv::Mat mat2) = 0;
};

class GenTopoWeight {
  public:
    GenTopoWeight() {}

    virtual ~GenTopoWeight() {}
    
    virtual cv::Mat1f GetWeight(const size_t &t) = 0;

};

}
