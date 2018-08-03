#include <iostream>
#include <memory>
#include <random>
#include <unistd.h>

#include <opencv2/opencv.hpp>

#include  "base/circleTopo.h"
#include "base/l2_match.h"
#include "base/exp_lr_decay.h"
#include "som.h"

#include "utils/initialization.h"

namespace SOM {
void testCircleTopo() {
  size_t R0 = 3, num_iters = 10;
  const std::shared_ptr<SOM::CircleTopoWeight> topoGenerator = std::make_shared<CircleTopoWeight>(R0, num_iters);
  for( size_t i=0; i < 6; ++i) {
    std::cout << "Iter:"<< i << std::endl;
    cv::Mat1f topoWeight = topoGenerator->GetWeight(i);
    std::cout << topoWeight << std::endl;
  }
}


void testMatcher() {
  auto rst1 = Matcher<float>::L2Match(cv::Mat_<float>::ones(2, 3), cv::Mat_<float>::zeros(4, 3), 2);
  auto rst2 = Matcher<int>::L2Match(cv::Mat_<int>::ones(2, 3), cv::Mat_<int>::zeros(4, 3), 2);
}


void test() {
  std::vector<float> a(20);
  std::iota(a.begin(), a.end(), 0);
  for(const auto &val : a) {
    std::cout << val << " ";
  }
  std::cout << std::endl;
  std::random_shuffle(a.begin(), a.end());
  for(const auto &val : a) {
    std::cout << val << " ";
  }
  std::cout << std::endl;

  cv::Mat1f mat1(a);
  mat1  = mat1.reshape(0, 4);
  std::cout << mat1 << std::endl;
  std::random_shuffle(mat1.begin(), mat1.end());
  std::cout << mat1.end() - mat1.begin() << std::endl;
  std::cout << mat1 << std::endl;
}

void testSOM() {
  size_t H = 3, W = 3;
  size_t R0 = 3, num_epoch = 10;
  float base_lr = 0.1;
  std::vector<float> tmpW(27);
  std::iota(tmpW.begin(), tmpW.end(), 0.0);
  cv::Mat1f weight = cv::Mat1f(tmpW).reshape(0, 9);
  //std::cout << "weight " << weight << std::endl;
  
  std::vector<float> tmpX(30);
  //std::iota(tmpX.begin(), tmpX.end(), 0);
  //cv::Mat1f X = cv::Mat1f(tmpX).reshape(0, 1);
  std::iota(tmpX.begin(), tmpX.end(), -10);
  cv::Mat1f X = cv::Mat1f(tmpX).reshape(0, 10);
  //cv::Mat1f X = cv::Mat1f::ones(1, 3);
  SOM2D som2d(H, W);
  som2d.InitTopoGenerator(R0, num_epoch);
  som2d.InitLRDecay(base_lr, num_epoch);
  som2d.Train(X, weight, 2, 10);
  std::cout << weight << std::endl;
}

void testLRExpDecay() {
  size_t num_iters = 10;
  float base_lr = 0.1;
  std::shared_ptr<LRExpDecay> lr_decay = std::make_shared<LRExpDecay>(base_lr, num_iters);
  for(size_t i = 0; i < num_iters; ++i) {
    std::cout << lr_decay->GetLR(i) << " " ;
  }
  std::cout << std::endl;
}


void testInit() {
  auto data =  fzy::init_by_normal(5);
  for(auto &val : data) {
    std::cout << val <<  " ";
  }
  std::cout << std::endl;
}
}

int main() {
// SOM::testCircleTopo();
//    SOM::testMatcher();
//    SOM::test();
//     SOM::testSOM();
//  SOM::testLRExpDecay();
  SOM::testInit();
}

