#include "som.h"
#include <sstream>
#include <fstream>
#include "utils/initialization.h"

namespace SOM {
cv::Mat readMatFromFile(const std::string &file) {
  size_t rows=0, cols = 0;
  std::ifstream fin(file);
  fin >> rows >> cols ;
  //std::cout << rows << " " << cols << std::endl;
  cv::Mat1f data(rows, cols);
  std::string line;
  for(size_t i=0; i < rows; ++i) {
    std::getline(fin, line);  
    std::stringstream ss;
    ss << line;
    for(size_t j=0; j < cols; ++j) {
      ss >> data.at<float>(i, j);
    }
  }
  fin.close();
  return data;
}

void trainSOM() {
  const size_t H = 400, W = 400, dims = 3;
  const size_t R0 = std::max(H, W) / 2, numEpoch = 200;
  const size_t batchSize = 20;
  const float base_lr = 1;
  /*
  cv::Mat1f X = (cv::Mat1f(8, dims)  << 1, 0, 0,
        0, 1, 0,
        0, 0, 1,
        0, 0.5, 0.25,
        0, 0, 0.5,
        1, 1, 0.2,
        1, 0.4, 0.25, 
        1, 0, 1 );
  */
  const std::string colorFile = "/data01/home/fanzhongyue/workspace/SOM/cpp/som/src/color_rbg.txt";
  cv::Mat1f X = readMatFromFile(colorFile);
  X /= 255.0;
  std::cout << X << std::endl;
//std::vector<float> weightVec = fzy::init_by_normal(H*W*dims);
  std::vector<float> weightVec = fzy::init_by_uniform(H*W*dims);
  cv::Mat1f weight = cv::Mat1f(weightVec).reshape(0, H*W);
  SOM2D som2d(H, W);
  som2d.InitTopoGenerator(R0, numEpoch);
  som2d.InitLRDecay(base_lr, numEpoch);
  som2d.Train(X, weight, numEpoch, batchSize);

  auto tmp = Matcher<float>::L2Match(weight, X, 1);
  sizeV idxs;
  for(auto &vec : tmp) {
    idxs.push_back(vec[0]);
  }
  cv::Mat img(H, W, CV_8UC3);
  for(size_t h=0; h < H; ++h) {
    for(size_t w=0; w < W; ++w) {
      auto idx = idxs[h*W + w];  
      unsigned char B = X.at<float>(idx, 0)*255;
      unsigned char G = X.at<float>(idx, 1)*255;
      unsigned char R = X.at<float>(idx, 2)*255;
      img.at<cv::Vec3b>(h, w) = cv::Vec3b(B, G, R);
    }
  }
  cv::imwrite("./som_c.jpg", img);
}
}

int main() {
  SOM::trainSOM();
}
