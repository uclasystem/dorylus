#ifndef __ADAM_HPP__
#define __ADAM_HPP__
#include <vector>
#include <cmath>
#include "../common/matrix.hpp"
#include "../common/utils.hpp"

const float BETA1 = .9;
const float BETA2 = .999;
const float EPSILON = 1e-07;

class AdamOptimizer {
  public:
    AdamOptimizer() {};
    ~AdamOptimizer();
    AdamOptimizer(float lr, std::vector<unsigned> dims);
    void nextIteration();
    void update(unsigned layer, FeatType *weight, FeatType *gradient);


  private:
    float learning_rate;
    unsigned epochs;
    std::vector<unsigned> dims;
    std::vector<FeatType *> momentum;
    std::vector<FeatType *> decay;

    float lr_t;
};


#endif
