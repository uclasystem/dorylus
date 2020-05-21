#include "../engine.hpp"

#include <omp.h>

FeatType* Engine::softmax(FeatType* inputTensor, unsigned rows, unsigned cols) {
    FeatType* result = new FeatType[rows * cols];

    // TODO: omp parallel for
    for (unsigned r = 0; r < rows; ++r) {
        FeatType* vecSrc = inputTensor + r * cols;
        FeatType* vecDst = result + r * cols;

        FeatType denom = 1e-20;
        FeatType maxElem = *(std::max_element(vecSrc, vecSrc + cols));
        for (unsigned c = 0; c < cols; ++c) {
            vecDst[c] = std::exp(vecSrc[c] - maxElem);
            denom += vecDst[c];
        }

        for (unsigned c = 0; c < cols; ++c) {
            vecDst[c] /= denom;
        }
    }

    return result;
}
