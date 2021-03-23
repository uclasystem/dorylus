#include "../engine.hpp"

#include <omp.h>


// FORWARD OPS
FeatType* Engine::softmax(FeatType* inputTensor, FeatType* result, unsigned rows, unsigned cols) {
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



// BACKWARD OPS
// Takes as input
Matrix Engine::softmax_prime(FeatType* valuesTensor, FeatType* softmaxOutput, unsigned size) {
    FeatType* result = new FeatType[size];

    for (unsigned long long eid = 0; eid < size; ++eid) {
        result[eid] = (softmaxOutput[eid] / valuesTensor[eid]) - softmaxOutput[eid];
    }

    return Matrix(size, 1, result);
}

// Elementwise multiplicaiton b/w sparse and dense matrices
// Sparse input tensor is any CSC value vector. Defaults to csc.values
Matrix Engine::sparse_dense_elemtwise_mult(CSCMatrix<EdgeType>& csc,
  FeatType* sparseInputTensor, FeatType* denseInputTensor) {
    if (sparseInputTensor == NULL) sparseInputTensor = csc.values;

    return Matrix();
}

/**
 *
 * Compute leakyReLU for an array (overwrites data)
 *
 */
FeatType* Engine::leakyReLU(FeatType* matxData, unsigned vecSize) {
    float alpha = .01;
    FeatType* result = new FeatType[vecSize];

    for (unsigned i = 0; i < vecSize; ++i) {
        result[i] = (matxData[i] > 0) ? matxData[i] : alpha * matxData[i];
    }

    return result;
}

/**
 *
 * Compute leakyReLU for a single value
 *
 */
FeatType Engine::leakyReLU(FeatType f) {
    float alpha = .01;
    return (f > 0) ? f : alpha * f;
}

// NOTE TO SELF:
// leakyReLU derivative:
//          { if x > 0 : x
// f'(x) =  {
//          { else .01


