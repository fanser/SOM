#include "circleTopo.h"

namespace SOM {
CircleTopoWeight::CircleTopoWeight(const size_t &R0, const size_t &num_iters):
  R0_(R0),
  num_iters_(num_iters),
  T0_(float(num_iters) / std::log(R0_ + eps_))  {
  }

cv::Mat1f CircleTopoWeight::GetWeight(const size_t &t) {
  float radius = GetRadius(t);
  cv::Mat1f topoWeight = GetWeight(radius);
  return topoWeight
}

float CircleTopoWeight::GetRadius(const size_t &t) {
  float radius = R0_ * std::exp(- t / T0_);    
  return radius;
}

cv::Mat1f CircleTopoWeight::GetWeight(const float &radius) {
  int r = std::round(radius);
  int H = W = 2*r + 1;
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
