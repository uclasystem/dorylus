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
#include <thrust/reduce.h>

//It maintains a GPU context
class ComputingUnit
{
public:
	ComputingUnit();
	Matrix dot(Matrix& A,Matrix& B); 	// use cublas for speed
	Matrix activate(const Matrix& A){}; //probably use thrust. Since its not a Linear Algebra op
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

Matrix ComputingUnit::dot(Matrix& A, Matrix& B){
    CuMatrix devA(A,handle);
    CuMatrix devB(B,handle);
    CuMatrix devC=devA.dot(devB);
    devC.updateMatrixFromGPU();
}



#endif