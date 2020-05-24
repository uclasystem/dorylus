#ifndef __ADAM_HPP__
#define __ADAM_HPP__
#include <vector>
#include <cmath>
#include "../common/matrix.hpp"
#include "../common/utils.hpp"

class AdamOptimizer {
  public:
    AdamOptimizer() {};
    ~AdamOptimizer();
    AdamOptimizer(float lr, std::vector<unsigned> dims);
    void nextIteration();
    void update(unsigned layer, FeatType *weight, FeatType *gradient);
    void setLR(float lr) { learning_rate = lr; };
    void decayAlpha(float decayRate) { BETA1 *= decayRate; };

    float BETA1 = .9;
    float BETA2 = .999;
    float EPSILON = 1e-07;

    // float WEIGHT_DECAY = 0.05; // or 5e-4?
    // const float WEIGHT_DECAY = 5e-4;
    const float WEIGHT_DECAY = 0;

  private:
    float learning_rate;
    unsigned epochs;
    std::vector<unsigned> dims;
    std::vector<FeatType *> momentum;
    std::vector<FeatType *> decay;

    float lr_t;
};


#endif
