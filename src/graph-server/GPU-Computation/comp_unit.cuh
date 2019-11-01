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



//TODO: modify function return value and signatures to fit the real workload
//		This is important because it can reduce memory copy from GPU and RAM
//AWARE: to free Matrix.data in long run
//It maintains a GPU context
class ComputingUnit
{
public:
	ComputingUnit();

    CuMatrix wrapMatrix(Matrix m);

	CuMatrix dot(Matrix& A,Matrix& B);
	void activate(CuMatrix& A);
    CuMatrix softmaxRows(CuMatrix &mat);
    CuMatrix hadamardSub(CuMatrix& matLeft,CuMatrix& matRight);
    CuMatrix hadamardMul(CuMatrix& matLeft, CuMatrix& matRight);
    CuMatrix activateDerivative(CuMatrix& mat);
    CuMatrix dotGDwithWTrans(CuMatrix& matLeft, CuMatrix& matRight);
    CuMatrix dotActTranswithGD(CuMatrix& matLeft, CuMatrix& matRight, const float learning_rate);

    unsigned checkAccuracy(CuMatrix& predictions, CuMatrix& labels);
	float checkLoss(CuMatrix& preds, CuMatrix& labels);
    
    ~ComputingUnit(){cublasDestroy(handle);}


// private:
    cudnnHandle_t cudnnHandle;
	cublasHandle_t handle;
	cublasStatus_t stat;
};


#endif