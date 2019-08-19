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
#include "../../../src/utils/utils.hpp"
#include "CuMatrix.hpp"

#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/transform.h>




//TODO: modify function return value and signatures to fit the real workload
//		This is important because it can reduce memory copy from GPU and RAM
//AWARE: to free Matrix.data in long run
//It maintains a GPU context
class ComputingUnit
{
public:
	ComputingUnit();
	CuMatrix dot(Matrix& A,Matrix& B); 	// use cublas for speed
	void activate(Matrix& A); 	//use thrust. Since its not a Linear Algebra op

	~ComputingUnit(){cublasDestroy(handle);}

private:
	cublasHandle_t handle;
	cublasStatus_t stat;
};





#endif