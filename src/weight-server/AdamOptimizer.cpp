#include "AdamOptimizer.hpp"
using std::isnan;
AdamOptimizer::AdamOptimizer(float lr, std::vector<unsigned> dims_) {
    learning_rate = lr;
    dims = dims_;
    epochs = 0;
    lr_t = 0;
    nextIteration();
    for (unsigned ui = 0; ui < dims.size() - 1; ++ui) {
        unsigned dataSize = dims[ui] * dims[ui + 1];
        FeatType *momentumptr = new FeatType[dataSize];
        FeatType *decayptr = new FeatType[dataSize];

        std::memset(momentumptr, 0, dataSize * sizeof(FeatType));
        std::memset(decayptr, 0, dataSize * sizeof(FeatType));

        momentum.push_back(momentumptr);
        decay.push_back(decayptr);
    }
}

AdamOptimizer::~AdamOptimizer() {
    for (unsigned i = 0; i < momentum.size(); ++i) {
        delete[] momentum[i];
        delete[] decay[i];
    }
}

void AdamOptimizer::nextIteration() {
    ++epochs;
    float beta_1_power = pow(BETA1, epochs);
    float beta_2_power = pow(BETA2, epochs);
    lr_t = learning_rate * (sqrt(1 - beta_2_power)) / (1 - beta_1_power);
}

void AdamOptimizer::update(unsigned layer, FeatType *weight, FeatType *gradient) {
    unsigned size = dims[layer] * dims[layer + 1];
    for (unsigned i = 0; i < size; ++i) {
        float gt = gradient[i] + WEIGHT_DECAY * weight[i];
        float prev_m = momentum[layer][i];
        float prev_d = decay[layer][i];
        momentum[layer][i] = BETA1 * prev_m + (1. - BETA1) * gt;
        decay[layer][i] = BETA2 * prev_d + (1. - BETA2) * gt * gt;
        float delta = lr_t * (momentum[layer][i]) / (sqrt(decay[layer][i]) + EPSILON);

        weight[i] -= delta;

    }
    if(layer == 0)
        nextIteration();
}
