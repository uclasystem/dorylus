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
#include "cublas_v2.h"
#include "cu_matrix.cuh"

#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/reduce.h>


//TODO: modify function return value and signatures to fit the real workload
//		This is important because it can reduce memory copy from GPU and RAM
//AWARE: to free Matrix.data in long run
//It maintains a GPU context
class ComputingUnit
{
public:
	ComputingUnit();

    CuMatrix wrapMatrix(Matrix m);

	CuMatrix dot(const Matrix& A,const Matrix& B); 	
	void activate(CuMatrix& A); 	
    CuMatrix softmaxRows(const CuMatrix &mat);
    CuMatrix hadamardSub(const CuMatrix& matLeft,const CuMatrix& matRight);
    CuMatrix* hadamardMul(const CuMatrix& matLeft,const CuMatrix& matRight);
    CuMatrix activateDerivate(const CuMatrix& mat);
    CuMatrix dotGDwithWTrans(const CuMatrix& matLeft,const CuMatrix& matRight);
    CuMatrix dotActTranswithGD(const CuMatrix& matLeft, const CuMatrix& matRight, const float learning_rate);
	~ComputingUnit(){cublasDestroy(handle);}
// private:
	cublasHandle_t handle;
	cublasStatus_t stat;
};


#endif