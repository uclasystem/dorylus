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
#include "../../../src/utils/utils.hpp"
#include "CuMatrix.hpp"

#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/transform.h>



struct tanh_functor
{
    tanh_functor(){}
    __host__ __device__
        float operator()(const float& x) const { return tanhf(x);}
};

struct exp_functor
{
    exp_functor(){}
    __host__ __device__
        float operator()(const float& x) const { return expf(x);}
};

//TODO: modify function return value and signatures to fit the real workload
//		This is important because it can reduce memory copy from GPU and RAM
//AWARE: to free Matrix.data in long run
//It maintains a GPU context
class ComputingUnit
{
public:
	ComputingUnit();
	CuMatrix dot(Matrix& A,Matrix& B); 	// use cublas for speed
	void activate(CuMatrix& A); 	//use thrust. Since its not a Linear Algebra op
    CuMatrix softmaxRows(Matrix &A);
	~ComputingUnit(){cublasDestroy(handle);}

// private:
	cublasHandle_t handle;
	cublasStatus_t stat;
};






ComputingUnit::ComputingUnit(){
    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS initialization failed\n");
        exit (EXIT_FAILURE);
    }
    cublasSetMathMode(handle,CUBLAS_TENSOR_OP_MATH);
}

CuMatrix ComputingUnit::softmaxRows(Matrix &mat){
    CuMatrix devMat(mat,handle);
    CuMatrix res(Matrix(mat.getRows(),mat.getCols(),new FeatType[mat.getNumElemts()]),handle);
    thrust::device_ptr<float> devMat_ptr(devMat.devPtr);
    thrust::device_ptr<float> res_ptr(res.devPtr);
    // thrust::transform(devMat_ptr, devMat_ptr+mat.getNumElemts(),res_ptr, exp_functor());
    float* row_sums=new FeatType[mat.getCols()];
    unsigned row_len=mat.getRows();
    for(int i=0;i<mat.getCols();++i){
        row_sums[i] =thrust::reduce(devMat_ptr+i*row_len,devMat_ptr+(i+1)*row_len);
        printf("%f\n",row_sums[i] );
    }
    // thrust::device_ptr<float> row_sums_ptr(row_sums.devPtr);
    printf("hi\n");
    delete[] row_sums;
    return res;
}

CuMatrix ComputingUnit::dot(Matrix& A, Matrix& B){
    CuMatrix devA(A,handle);
    CuMatrix devB(B,handle);
    CuMatrix devC=devA.dot(devB);
    devC.updateMatrixFromGPU();
    return devC;
}

void ComputingUnit::activate(CuMatrix& A){
	thrust::device_ptr<float> devA_ptr(A.devPtr);
  	thrust::transform(devA_ptr, devA_ptr+A.getNumElemts(),devA_ptr, tanh_functor());
  	A.updateMatrixFromGPU();
}




#endif