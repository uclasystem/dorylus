#ifndef __COMP_UNIT_HPP__
#define __COMP_UNIT_HPP__

#include <chrono>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <cmath>
#include <cstring>
// #include <cstdlib>
// #include <cmath>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "../../src/utils/utils.hpp"
#include <memory>



float* deviceMalloc(const Matrix& M){
	float* devPtr;
	cudaError_t cudaStat;
    cudaStat = cudaMalloc ((void**)&devPtr, M.rows*M.cols*sizeof(float));
    if (cudaStat != cudaSuccess) {
        printf ("device memory allocation failed");
        exit (EXIT_FAILURE);
    }
    return devPtr;
}
void deviceSetMatrix(cublasHandle_t &handle,const Matrix& M, float* devPtr ){
	cublasStatus_t stat;
	stat = cublasSetMatrix (M.rows, M.cols, sizeof(float), M.data, M.rows , devPtr, M.rows);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("data download failed");
        cudaFree (devPtr);
        cublasDestroy(handle);
        exit (EXIT_FAILURE);
    }
}

void deviceFetchMatrix(float* devPtr,Matrix M){
	cublasStatus_t stat = cublasGetMatrix (M.rows, M.cols, sizeof(float), devPtr, M.rows, M.data, M.rows );
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("data upload failed");
        cudaFree (devPtr);
        // cublasDestroy(handle);
        exit (EXIT_FAILURE);
    }
}

class ComputingUnit
{
public:
	ComputingUnit(){};
	Matrix dot(Matrix& A,Matrix& B); 	// use cublas for speed
	Matrix activate(const Matrix& A); //probably use thrust. Since its not a Linear Algebra op
	~ComputingUnit(){};
};

Matrix ComputingUnit::dot(Matrix& A, Matrix& B){
    cublasHandle_t handle;
    Matrix C(A.rows,B.cols,new float[A.rows*B.cols]);
    float* devPtrA=deviceMalloc(A);
    float* devPtrB=deviceMalloc(B);
    float* devPtrC=deviceMalloc(C);
   	
   	cublasStatus_t stat;
    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS initialization failed\n");
        exit (EXIT_FAILURE);
    }
    
    deviceSetMatrix(handle,A,devPtrA);
    deviceSetMatrix(handle,B,devPtrB);
    deviceSetMatrix(handle,C,devPtrC);

    float alpha=1.0f;
    float beta=0.0f;
    //1. cublas is using col-major
    //2. when cpy into/out device memory, it will do Transpose 
    //3. C=AB and C^T= (B^T*A^T)
    //This means just swap the order of multiplicaiton
    //Guide: https://peterwittek.com/cublas-matrix-c-style.html
   	cublasSgemm(handle,
   		CUBLAS_OP_N,CUBLAS_OP_N,
    	B.cols,A.rows,A.cols,
    	&alpha,
    	devPtrB,B.cols,
    	devPtrA,A.cols,
    	&beta,
    	devPtrC,B.cols);

    stat = cublasGetMatrix (C.rows, C.cols, sizeof(float), devPtrC, C.rows, C.data, C.rows );
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("data upload failed");
        cudaFree (devPtrC);
        cublasDestroy(handle);
        exit (EXIT_FAILURE);
    }
    // std::cout<<(C.str());
    cudaFree (devPtrA);
    cudaFree (devPtrB);
	cudaFree (devPtrC);


}


#endif