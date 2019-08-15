#ifndef __COMP_UNIT_HPP__
#define __COMP_UNIT_HPP__

#include <cstring>
#include <chrono>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <cmath>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "../../src/utils/utils.hpp"
#include "CuMatrix.hpp"

#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/copy.h>

struct act_functor
{
    act_functor(){}
    __host__ __device__
        float operator()(const float& x) const { return std::tanh(x);}
};

//It maintains a GPU context
class ComputingUnit
{
public:
	ComputingUnit();
	CuMatrix dot(Matrix& A,Matrix& B); 	// use cublas for speed
	// void activate(CuMatrix& A); //probably use thrust. Since its not a Linear Algebra op
	void activate(Matrix& A);

	~ComputingUnit(){cublasDestroy(handle);}

private:
	cublasHandle_t handle;
	cublasStatus_t stat;
};

ComputingUnit::ComputingUnit(){
    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS initialization failed\n");
        exit (EXIT_FAILURE);
    }
}

CuMatrix ComputingUnit::dot(Matrix& A, Matrix& B){
    CuMatrix devA(A,handle);
    CuMatrix devB(B,handle);
    CuMatrix devC=devA.dot(devB);
    devC.updateMatrixFromGPU();
    return devC;
}

void ComputingUnit::activate(Matrix& A){
	CuMatrix devA(A,handle);
	thrust::device_ptr<float> devA_ptr(devA.devPtr);
  	thrust::transform(devA_ptr, devA_ptr+A.rows*A.cols,devA_ptr, act_functor());
  	devA.updateMatrixFromGPU();
  	cout<<devA.str();
}




#endif