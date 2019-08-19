#include "comp_unit.hpp"

struct act_functor
{
    act_functor(){}
    __host__ __device__
        float operator()(const float& x) const { return std::tanh(x);}
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
  	thrust::transform(devA_ptr, devA_ptr+A.getRows()*A.getCols(),devA_ptr, act_functor());
  	devA.updateMatrixFromGPU();
    printf("%s\n", devA.str().c_str());

}
