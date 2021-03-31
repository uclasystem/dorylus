#ifndef __COMP_UNIT_HPP__
#define __COMP_UNIT_HPP__

#include <cuda_runtime.h>
#include <cudnn.h>
#include <cusparse.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>

#include "../utils/utils.hpp"
#include "cu_matrix.cuh"
#include "cublas_v2.h"

// AWARE: to free Matrix.data in long run
// It maintains a GPU context
class ComputingUnit {
   public:
    static ComputingUnit &getInstance();

    CuMatrix leakyRelu(CuMatrix &m, float coef);
    CuMatrix leakyReluPrime(CuMatrix &m, float coef);
    CuMatrix reduceColumns(CuMatrix m);
    CuMatrix gatherRows(CuMatrix m, std::vector<int> indices);
    CuMatrix gatherRowsGthr(CuMatrix m, int *indices,int len);

    CuMatrix wrapMatrix(Matrix m);

    void scaleRowsByVector(CuMatrix &CuM, CuMatrix &cuV);
    CuMatrix aggregate(CuMatrix &sparse, CuMatrix &dense, CuMatrix &norms);

    CuMatrix dot(Matrix &A, Matrix &B);
    void activate(CuMatrix &A);
    CuMatrix softmaxRows(CuMatrix &mat);
    void hadamardAdd(CuMatrix &matLeft, CuMatrix &matRight);
    CuMatrix hadamardSub(CuMatrix &matLeft, CuMatrix &matRight);
    CuMatrix hadamardMul(CuMatrix &matLeft, CuMatrix &matRight);
    CuMatrix activateBackward(CuMatrix &x, CuMatrix &y, CuMatrix &dy);

    unsigned checkAccuracy(CuMatrix &predictions, CuMatrix &labels);
    float checkLoss(CuMatrix &preds, CuMatrix &labels);
    void getTrainStat(CuMatrix &preds, CuMatrix &labels, float &acc,
                      float &loss);
    void maskout(CuMatrix &preds, CuMatrix &labels);

    cudnnHandle_t cudnnHandle;
    cusparseHandle_t spHandle;
    cublasHandle_t handle;
    cublasStatus_t stat;
    cudaStream_t stream;

   private:
    ComputingUnit();
    static ComputingUnit *instance;
};

// global variables
extern CuMatrix *NormAdjMatrixIn;
extern CuMatrix *NormAdjMatrixOut;
extern CuMatrix *OneNorms;
extern CuMatrix *ZeroNorms;
extern ComputingUnit cu;

#endif