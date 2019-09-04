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
#include <thrust/iterator/discard_iterator.h>
#include <thrust/reduce.h>


struct tanh_functor
{
    tanh_functor(){}
    __host__ __device__
        float operator()(const float& x) const { return tanhf(x);}
};

struct activateDerivate_functor
{
    activateDerivate_functor(){}
    __host__ __device__
        float operator()(const float& x) const { 
            return 1 - pow(tanh(x), 2);
        }
};

struct exp_functor
{
    exp_functor(){}
    __host__ __device__
        float operator()(const float& x) const { return expf(x);}
};

struct divide_functor
{
    divide_functor(float *denoms_):denoms(denoms_){}
    __host__ __device__
        float operator()(const int& i,const float& x) const { return x/denoms[i];}
    float* denoms;
};

struct setRow_functor
{
    setRow_functor(unsigned col_):col(col_){}
    unsigned col;
    __host__ __device__
        int operator()(const int& x) const { return x/col;}
};


//TODO: modify function return value and signatures to fit the real workload
//		This is important because it can reduce memory copy from GPU and RAM
//AWARE: to free Matrix.data in long run
//It maintains a GPU context
class ComputingUnit
{
public:
	ComputingUnit();
	CuMatrix dot(Matrix& A,Matrix& B); 	
	void activate(CuMatrix& A); 	
    // Matrix softmaxRows(Matrix &mat);
    CuMatrix softmaxRows(CuMatrix &mat);
    CuMatrix hadamardSub(CuMatrix& matLeft, CuMatrix& matRight);
    CuMatrix hadamardMul(CuMatrix& matLeft, CuMatrix& matRight);
    CuMatrix activateDerivate(CuMatrix& mat);
    CuMatrix dotGDwithWTrans(CuMatrix& matLeft, CuMatrix& matRight);
    CuMatrix dotActTranswithGD(CuMatrix& matLeft, CuMatrix& matRight, float learning_rate);
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



CuMatrix ComputingUnit::hadamardSub(CuMatrix& matLeft, CuMatrix& matRight) {

    CuMatrix res(Matrix(matLeft.getRows(),matLeft.getCols(), new FeatType[matLeft.getNumElemts()]),handle);

    thrust::device_ptr<float> cuLeft_ptr(matLeft.devPtr);
    thrust::device_ptr<float> cuRight_ptr(matRight.devPtr);
    thrust::device_ptr<float> res_ptr(res.devPtr);

    thrust::transform(cuLeft_ptr,cuLeft_ptr+matLeft.getNumElemts(),
                        cuRight_ptr,
                        res_ptr,
                        thrust::minus<float>());

    return res;
}

CuMatrix ComputingUnit::hadamardMul(CuMatrix& matLeft, CuMatrix& matRight) {

    CuMatrix res(Matrix(matLeft.getRows(),matLeft.getCols(), new FeatType[matLeft.getNumElemts()]),handle);

    thrust::device_ptr<float> cuLeft_ptr(matLeft.devPtr);
    thrust::device_ptr<float> cuRight_ptr(matRight.devPtr);
    thrust::device_ptr<float> res_ptr(res.devPtr);

    thrust::transform(cuLeft_ptr,cuLeft_ptr+matLeft.getNumElemts(),
                        cuRight_ptr,
                        res_ptr,
                        thrust::multiplies<float>());

    return res;
}

// GPU is faster than CPU when SIZE>250000, Maybe CPU is faster in most cases 
CuMatrix ComputingUnit::softmaxRows(CuMatrix &mat){
    CuMatrix res(Matrix(mat.getRows(),mat.getCols(),new FeatType[mat.getNumElemts()]),handle);
    thrust::device_ptr<float> devMat_ptr(mat.devPtr);
    thrust::device_ptr<float> res_ptr(res.devPtr);
    thrust::transform(devMat_ptr, devMat_ptr+mat.getNumElemts(),res_ptr, exp_functor());
   
    thrust::device_vector<int> indices(mat.getNumElemts());
    thrust::counting_iterator<int> idxfirst(0);
    thrust::counting_iterator<int> idxlast = idxfirst +mat.getNumElemts();
    thrust::transform(idxfirst, idxlast, indices.begin(), setRow_functor(mat.getCols()));
    thrust::device_vector<float> row_sums(mat.getRows());
    thrust::reduce_by_key(indices.begin(), indices.end(),
                        res_ptr,
                        thrust::make_discard_iterator(),
                        row_sums.begin());
    thrust::transform(indices.begin(), indices.end(),res_ptr,res_ptr, divide_functor(thrust::raw_pointer_cast(row_sums.data())));

    return res;
}

CuMatrix ComputingUnit::activateDerivate(CuMatrix& mat) {
    CuMatrix res(Matrix(mat.getRows(),mat.getCols(),new FeatType[mat.getNumElemts()]),handle);
    thrust::device_ptr<float> res_ptr(res.devPtr);
    CuMatrix z(Matrix(mat.getRows(),mat.getCols(),mat.getData()),handle);
    thrust::device_ptr<float> z_ptr(z.devPtr);
    thrust::transform(z_ptr, z_ptr+mat.getNumElemts(),res_ptr, activateDerivate_functor());
    return res;
}


CuMatrix
ComputingUnit::dotGDwithWTrans(CuMatrix& matLeft, CuMatrix& matRight) {
    CuMatrix res=matLeft.dot(matRight,1.0f,0.f,false,true);
    return res;
}

CuMatrix
ComputingUnit::dotActTranswithGD(CuMatrix& matLeft, CuMatrix& matRight, float learning_rate) {
    CuMatrix res=matLeft.dot(matRight,learning_rate,0,true,false);
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