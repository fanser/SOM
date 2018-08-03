#pragma once
#include "base.h"

namespace SOM {
class CircleTopoWeight : public GenTopoWeight {
public:
 //CircleTopoWeight() { CircleTopoWeight(3, 1000);}

  CircleTopoWeight(const size_t &R0, const size_t &num_iters);

  virtual cv::Mat GetWeight(const size_t &t) override;

private:
  cv::Mat1f GetWeight(const float &radius);
  float GetRadius(const size_t &t);

private:
  const size_t num_iters_;
  const float eps_ = 1e-6;
  const size_t R0_;
  const float T0_;

};


}

