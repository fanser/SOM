#pragma once
#include <vector>

#include <opencv2/opencv.hpp>

namespace SOM {
class GenTopoWeight {
  public:
    GenTopoWeight() {}

    virtual ~GenTopoWeight() {}
    
    virtual cv::Mat GetWeight(const size_t &t) = 0;

};

class LRDecay {
public:
  LRDecay(const float &base_lr, const size_t num_iters):
    base_lr_(base_lr),
    num_iters_(num_iters) {
    }

  virtual float GetLR(const size_t &iter) = 0;

protected:
  const float base_lr_;
  const size_t num_iters_;

};

}
