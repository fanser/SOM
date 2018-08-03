#pragma once

#include "base.h" 

namespace SOM {
class LRExpDecay : public LRDecay {
public:
  LRExpDecay(const float &base_lr, const size_t &num_iters) :
    LRDecay(base_lr, num_iters) {
    }

  virtual float GetLR(const size_t &iter) override{
    float lr = base_lr_ * std::exp( -float(iter) / num_iters_);
    return lr;
  }

};

}
