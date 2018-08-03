#pragma once
#include <memory>

#include "base/circleTopo.h"
#include "base/l2_match.h"
#include "base/exp_lr_decay.h"

namespace SOM{
class SOM2D {
public:
  SOM2D(const size_t &h, const size_t &w);

  void InitTopoGenerator(const size_t &R0, const size_t &num_iters) {
    topoGenerator_.reset(new CircleTopoWeight(R0, num_iters));
  }

  void InitLRDecay(const float &base_lr, const size_t &num_iters) {
    lrDecayer_.reset(new LRExpDecay(base_lr, num_iters)); 
  }

  void Train(const cv::Mat1f &X, cv::Mat1f &W, const size_t &numEpoch, const size_t &batchSize=1);

  cv::Mat1f ShuffleDset(const cv::Mat1f &X);

private:
  cv::Mat1f CalcDeltaWeight(const cv::Mat1f &topoW, const cv::Mat1f &X, const cv::Mat1f &W, const sizeV &winner);

  //cv::Mat1f ShuffleDset(const cv::Mat1f &X);

private:
  const size_t H_, W_;

  std::shared_ptr<GenTopoWeight> topoGenerator_;

  std::shared_ptr<LRDecay> lrDecayer_;

}
;

}
