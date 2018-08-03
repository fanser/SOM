#pragma once
#include <vector>

#include <opencv2/opencv.hpp>

namespace SOM {
typedef std::vector<size_t> sizeV;

template<class T>
class Matcher {
public: 
    static std::vector<sizeV> L2Match(const cv::Mat_<T> &mat1, const cv::Mat_<T> &mat2, const size_t &topk=1) {
      return std::move(_L2Match(mat1, mat2, topk));
    }

    //static std::vector<std::vector<size_t> > L1Match(const cv::Mat_<T> &mat1, const cv::Mat_<T> &mat2); 

    //static std::vector<std::vector<size_t> > HMMatch(const cv::Mat_<T> &mat1, const cv::Mat_<T> &mat2); 

    //static std::vector<std::vector<size_t> > CosineMatch(const cv::Mat_<T> &mat1, const cv::Mat_<T> &mat2); 

private:
    static std::vector<sizeV> _L2Match(const cv::Mat &mat1, const cv::Mat &mat2, const size_t &topk); 

};

template <class T> 
std::vector<sizeV> Matcher<T>::_L2Match(const cv::Mat &mat1, const cv::Mat &mat2, const size_t &topk) {
  size_t num_idxs = std::min((int)topk, mat2.cols);
  //std::cout << mat1.rows << " " << mat1.cols << " " << mat2.rows << " " << mat2.cols << std::endl;
  std::vector<sizeV> indexs;
  std::vector<T> dists(mat2.rows);
  for(int i=0; i < mat1.rows; ++i) {
    dists.clear();
    for(int j=0; j < mat2.rows; ++j) {
      T dist{};
      for(int k=0; k < mat2.cols; ++k) {
        auto tmp = mat1.at<T>(i, k) - mat2.at<T>(j, k);
        dist += tmp * tmp;
      }
      dists.push_back(dist);
    }
    sizeV row_idxs(mat2.rows);
    auto begin_iter = row_idxs.begin(), end_iter = row_idxs.end();
    std::iota(begin_iter, end_iter, 0);
    std::nth_element(begin_iter, begin_iter + num_idxs, end_iter, 
      [&dists](const size_t &idx1, const size_t &idx2) {
          return dists[idx1] < dists[idx2];
      }
      );
    /*
    std::cout << "idxs size" << row_idxs.size() << std::endl;
    for(size_t i=0; i < row_idxs.size(); ++i) {
      std::cout <<  dists[row_idxs[i]] << " ";
    }
    row_idxs.resize(num_idxs);
    std::cout << std::endl;
    */
    indexs.push_back(row_idxs);
  }
  return std::move(indexs);
}

}
