#include "circleTopo.h"

namespace SOM {
CircleTopoWeight::CircleTopoWeight(const size_t &R0, const size_t &num_iters):
  R0_(R0),
  num_iters_(num_iters),
  T0_(float(num_iters) / std::log(R0_ + eps_))  {
    std::cout << "R0: " << R0_ << std::endl;
    std::cout << "T0: " << T0_ << std::endl;
  }

float CircleTopoWeight::GetRadius(const size_t &t) {
  float radius = R0_ * std::exp(- static_cast<float>(t) / T0_);    
  //std::cout << "radius " << radius << std::endl;
  return radius;
}

cv::Mat CircleTopoWeight::GetWeight(const size_t &t) {
  float radius = GetRadius(t);
  //std::cout << "radius " << radius << std::endl;
  cv::Mat1f topoWeight = GetWeight(radius);
  return topoWeight;
}


cv::Mat1f CircleTopoWeight::GetWeight(const float &radius) {
  int r = int(radius);
  int H = 2*r + 1, W = H;
  cv::Mat1f topoWeight = cv::Mat1f::zeros(H, W);
  float r_sq = radius * radius;
  for(int h=0; h < H; ++h) {
    for(int w=0; w < W; ++w) {
      float dist_sq = (h - r) * (h -r ) + (w -r) * (w -r);
      if(dist_sq <= r_sq) {
        topoWeight.at<float>(h, w) = std::exp(- dist_sq / (2 * r_sq));
      }
    }
  }
  return topoWeight;
}

}
