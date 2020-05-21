#ifndef __CUMATIX_HPP__
#define __CUMATIX_HPP__

#include "cublas_v2.h"
#include <sstream>
#include <chrono>
#include <cstring>
#include <iostream>
#include <set>
#include <map>
#include "../../common/matrix.hpp"
#include "../../common/utils.hpp"
#include "../graph/graph.hpp"
#include "cusparse.h"


class CuMatrix : public Matrix {
  public:
    //for simple memory management
    static std::set<FeatType *> MemoryPool;
    static void freeGPU();

    CuMatrix() {setData(NULL);};
    CuMatrix(Matrix M, const cublasHandle_t &handle_);
    ~CuMatrix();

    void loadSpCSR(cusparseHandle_t &handle, Graph& graph);
    void loadSpCSC(cusparseHandle_t &handle, Graph& graph);

    void loadSpDense(FeatType *vtcsTensor, FeatType *ghostTensor,
                     unsigned numLocalVertices, unsigned numGhostVertices,
                     unsigned numFeat);

    CuMatrix extractRow(unsigned row);
    Matrix getMatrix();
    void updateMatrixFromGPU();
    void scale(const float &alpha);
    CuMatrix dot(CuMatrix &B, bool A_trans = false, bool B_trans = false, float alpha = 1., float beta = 0.);
    CuMatrix transpose();

    void deviceMalloc();
    void deviceSetMatrix();

    cublasHandle_t handle;
    cudaError_t cudaStat;
    cublasStatus_t stat;

    bool isSparse;
    // For normal matrix
    float *devPtr;

    //For sparse matrix
    //the memory will live throughout the lifespan of the program (I don't release them)
    unsigned long long nnz;
    EdgeType *csrVal;
    unsigned *csrColInd;
    unsigned *csrRowPtr;
};

#endif