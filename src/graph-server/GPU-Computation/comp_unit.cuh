#ifndef __COMP_UNIT_HPP__
#define __COMP_UNIT_HPP__

#include <cstring>
#include <chrono>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <cusparse.h>
#include "cublas_v2.h"
#include "cu_matrix.cuh"
#include "../utils/utils.hpp"


#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/reduce.h>
#include <thrust/extrema.h>
#include <thrust/iterator/counting_iterator.h>

//AWARE: to free Matrix.data in long run
//It maintains a GPU context
class ComputingUnit {
  public:
    static ComputingUnit &getInstance();

    CuMatrix wrapMatrix(Matrix m);

    CuMatrix scaleRowsByVector(Matrix m, Matrix v);
    CuMatrix aggregate(CuMatrix &sparse, CuMatrix &dense);

    CuMatrix dot(Matrix &A, Matrix &B);
    void activate(CuMatrix &A);
    CuMatrix softmaxRows(CuMatrix &mat);
    CuMatrix hadamardSub(CuMatrix &matLeft, CuMatrix &matRight);
    CuMatrix hadamardMul(CuMatrix &matLeft, CuMatrix &matRight);
    CuMatrix activateBackward(CuMatrix &y, CuMatrix &gradient);

    unsigned checkAccuracy(CuMatrix &predictions, CuMatrix &labels);
    float checkLoss(CuMatrix &preds, CuMatrix &labels);

    cudnnHandle_t cudnnHandle;
    cusparseHandle_t spHandle;
    cublasHandle_t handle;
    cublasStatus_t stat;

  private:
    ComputingUnit();
    static ComputingUnit *instance;
};


#endif