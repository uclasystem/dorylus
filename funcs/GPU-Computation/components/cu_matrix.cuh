#ifndef __CUMATIX_HPP__
#define __CUMATIX_HPP__

#include "cublas_v2.h"
#include <sstream> 
#include <chrono>
#include <cstring>
#include <iostream>
#include "../../../src/utils/utils.hpp"


class CuMatrix : public Matrix
{
public:
	CuMatrix(){};
    CuMatrix( Matrix M, const cublasHandle_t & handle_); 	
	~CuMatrix(); 

    Matrix getMatrix();
	void updateMatrixFromGPU();
    
	CuMatrix dot(const CuMatrix& M,float alpha=1.,float beta=0.) const ;
    CuMatrix transpose() const;
// private:
	void deviceMalloc();
	void deviceSetMatrix();

	cublasHandle_t handle;
	cudaError_t cudaStat;
	cublasStatus_t stat;
	float * devPtr;
};


#endif