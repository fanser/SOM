#include <iostream>
#include <memory>

#include  "base/circleTopo.h"

namespace SOM {
void testCircleTopo() {
  size_t R0 = 2, num_iters = 100;
  std::shared_ptr<SOM::CircleTopoWeight> topoGenerator = std::make_shared<CircleTopoWeight>(R0, num_iters);
  size_t iter = 1;
  cv::Mat1f topoWeight = topoGenerator->GetWeight(iter);
  std::cout << topoWeight << std::endl;
}

}

int main() {
  SOM::testCircleTopo();
}

