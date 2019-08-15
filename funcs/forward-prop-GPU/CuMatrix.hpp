#ifndef __CUMATIX_HPP__
#define __CUMATIX_HPP__
#include "cublas_v2.h"
#include "../../src/utils/utils.hpp"
#include <cstring>
#include <memory>

using std::endl;
using std::cout;
class CuMatrix : public Matrix
{
public:
	CuMatrix(){};
	CuMatrix(const CuMatrix& M);
	CuMatrix(const Matrix& M, cublasHandle_t & handle_);	
	~CuMatrix(); //will run for a long time. It's better to collect memory
	void updateMatrixFromGPU();
	CuMatrix dot(const CuMatrix& M);

	// friend class ComputingUnit; 

// private:
	void deviceMalloc();
	void deviceSetMatrix();

	cublasHandle_t handle;
	cudaError_t cudaStat;
	cublasStatus_t stat;
	float * devPtr;
};

CuMatrix::CuMatrix(const Matrix& M, cublasHandle_t & handle_)
	:Matrix(M.rows,M.cols,M.data){
	handle=handle_;
	deviceMalloc();
	deviceSetMatrix();
}

void CuMatrix::deviceMalloc(){
    cudaStat = cudaMalloc ((void**)&devPtr, rows*cols*sizeof(float));
    if (cudaStat != cudaSuccess) {
        printf ("device memory allocation failed\n");
        exit (EXIT_FAILURE);
    }
}
void CuMatrix::deviceSetMatrix(){
	stat = cublasSetMatrix (rows,cols, sizeof(float), data, rows , devPtr, rows);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("data download failed\n");
        cudaFree (devPtr);
        cublasDestroy(handle);
        exit (EXIT_FAILURE);
    }
}

void CuMatrix::updateMatrixFromGPU(){
	stat = cublasGetMatrix (rows, cols, sizeof(float), devPtr, rows, data, rows );
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("data upload failed\n");
        cudaFree (devPtr);
        cublasDestroy(handle);
        exit (EXIT_FAILURE);
    }
}

CuMatrix::~CuMatrix(){
	cudaFree (devPtr);
	delete[] data; 
}

CuMatrix CuMatrix::dot(const CuMatrix& M){
	if(handle!=M.handle){
		std::cout<<"Handle don't match\n";
		exit(EXIT_FAILURE);
	}
	Matrix mat_C=Matrix(rows,M.cols,new float[rows*M.cols*sizeof(float)]);
	CuMatrix C(mat_C,handle);
	float alpha=1.0f;
    float beta=0.0f;
    //1. cublas is using col-major
    //2. when cpy into/out device memory, it will do Transpose 
    //3. C=AB and C^T= (B^T*A^T)
    //This means just swap the order of multiplicaiton
    //Guide: https://peterwittek.com/cublas-matrix-c-style.html
	cublasSgemm(handle,
   		CUBLAS_OP_N,CUBLAS_OP_N,
    	M.cols,rows,cols,
    	&alpha,
    	M.devPtr,M.cols,
    	devPtr,cols,
    	&beta,
    	C.devPtr,M.cols);
	return C;
}



#endif