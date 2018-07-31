#pragma once
#include  <opencv2/opencv.hpp>

namespace SOM {
class GenTopoWeight {
  public:
    GenTopoWeight() {}

    virtual ~GenTopoWeight() {}
    
    virtual cv::Mat1f GetWeight(const size_t &t) = 0;

};

}
