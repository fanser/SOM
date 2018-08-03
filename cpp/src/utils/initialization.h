#pragma once
#include <random>
#include <algorithm>
#include <vector>
#include <ctime>

namespace fzy {

std::vector<float> init_by_normal(const size_t &size) {
  std::normal_distribution<float> distribution(0, 0.5);
  std::default_random_engine generator(std::time(0));
  std::vector<float> data(size);
  std::generate(data.begin(), data.end(), [&distribution, &generator]() {
    return distribution(generator);      
    });
  return std::move(data);
}

std::vector<float> init_by_uniform(const size_t &size) {
  int bound = 1000;
  std::uniform_int_distribution<int> distribution(-bound, bound);
  std::default_random_engine generator(std::time(0));
  std::vector<float> data(size);
  std::generate(data.begin(), data.end(), [&]() {
    return distribution(generator) / float(bound);      
    });
  return std::move(data);


}


}
