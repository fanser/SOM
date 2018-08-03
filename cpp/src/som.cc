#include "som.h"
#include <assert.h>
#include <ctime>

namespace SOM {
template<class T>
void printVec(const std::vector<T> &vec) {
  std::cout << "Print vector, size " << vec.size() << std::endl;
  std::cout << "*******************" << std::endl;
  for(const T &val : vec){
    std::cout << val << " ";
  }
  std::cout << "*******************" << std::endl;
}
    
SOM2D::SOM2D(const size_t &h, const size_t &w) :
  H_(h),
  W_(w) {
}

void SOM2D::Train(const cv::Mat1f &X, cv::Mat1f &W, const size_t &numEpoch, const size_t &batchSize) {
  assert(X.cols == W.cols);
  assert(X.rows >= 1 && W.rows >= 1);
  assert(H_ * W_ == W.rows);
  size_t numIters = std::ceil(float(X.rows) / batchSize);
  for(size_t epoch = 0; epoch < numEpoch; ++epoch) {
    auto XShuffled = ShuffleDset(X);
    //auto XShuffled = X;
    auto topoW = topoGenerator_->GetWeight(epoch);
    float lr = lrDecayer_->GetLR(epoch);
    //std::cout << "lr is " << lr << std::endl;
    for(size_t t = 0; t < numIters; ++t) {
      size_t batchStart = t*batchSize, batchEnd = std::min((t+1) * batchSize, (size_t)X.rows);
      auto XBatch = XShuffled.rowRange(batchStart, batchEnd);
      std::vector<sizeV> tmp = Matcher<float>::L2Match(XBatch, W);
      sizeV winner;
      for(auto &idxs : tmp) {
        winner.push_back(idxs[0]);
      }
      auto deltaW = CalcDeltaWeight(topoW, XBatch, W, winner);
      deltaW *= lr;
      W += deltaW;
      /*
      printVec(winner);
      std::cout << "Epoch " << epoch << " Iter: " << t << std::endl;
      std::cout << "T , topo weight " << t << topoW << std::endl;
      std::cout << "deltaW : " << deltaW << std::endl;
      std::cout << "W : " << W << std::endl;
      */
      std::cout << "Epoch: " << epoch << "/" << numEpoch << " Iter: " << t << "/" << numIters << std::endl;
    }
  }
}

cv::Mat1f SOM2D::CalcDeltaWeight(const cv::Mat1f &topoGraphy, const cv::Mat1f &X, const cv::Mat1f &W, const sizeV &winners) {
  assert(winners.size() == X.rows);
  cv::Mat1f deltaW = cv::Mat1f::zeros(H_*W_, W.cols); 
  cv::Mat1f mask = cv::Mat1f::ones(1, H_*W_);
  size_t topoH = (topoGraphy.rows - 1) / 2;
  size_t topoW = (topoGraphy.cols - 1) / 2;
  //std::cout << "topoH, topoW " << topoH << " "<< topoW << std::endl;
  for(size_t i=0; i < winners.size(); ++i) {
    size_t winner = winners[i];
    int h = winner / W_, w = winner % W_;
    int up = h - topoH, down = h + topoH + 1;
    int topoUp = 0, topoDown = 2*topoH + 1;
    if(up < 0)
      up = 0, topoUp = topoH - h;
    if(down > H_)
      down = H_, topoDown = topoH + H_ - h;
    
    int left = w - topoW, right = w + topoW + 1;
    int topoLeft = 0, topoRight = 2*topoW + 1;
    if(left < 0)
      left = 0, topoLeft = topoW - w;
    if(right > W_)
      right = W_, topoRight = topoW + W_ - w;
    /*
    std::cout << "topo size" <<topoH << " " << topoW << std::endl;
    std::cout << "up, down " << up << " "<< down << std::endl;
    std::cout << "left, right " << left << " "<< right << std::endl;
    std::cout << "topo up, down " << topoUp << " "<< topoDown << std::endl;
    std::cout << "topo left, right " << topoLeft << " "<< topoRight << std::endl;
    */
    assert(right - left == topoRight - topoLeft);
    assert(up - down == topoUp - topoDown);
    for(int j=0; j < down - up; ++j) {
      int y = up + j, topo_y = topoUp + j;
      for(int k=0; k < right - left; ++k) {
        int x = left + k, topo_x = topoLeft + k;
        deltaW.row(y*W_ + x) += topoGraphy.at<float>(topo_y, topo_x)
                                * (X.row(int(i)) - W.row(y*W_ + x));
        mask.at<float>(0, y*W_ + x) += 1.0;
      }
    } 
  }
  for(size_t i=0; i < H_ * W_; ++i) {
    deltaW.row(i) /= mask.at<float>(0, i);
  }
  return deltaW;
}

cv::Mat1f SOM2D::ShuffleDset(const cv::Mat1f &X) {
  sizeV idxs(X.rows);
  std::iota(idxs.begin(), idxs.end(), 0);
  std::shuffle(idxs.begin(), idxs.end(), std::default_random_engine(std::time(0)));
  //std::random_shuffle(idxs.begin(), idxs.end());

  cv::Mat1f X_shuffled = cv::Mat1f::zeros(X.rows, X.cols);
  for(size_t i=0; i < idxs.size(); ++i) {
    X_shuffled.row(i) += X.row(idxs[i]);
  }
  return X_shuffled;
}
}
