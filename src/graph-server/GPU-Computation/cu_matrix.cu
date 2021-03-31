#include "cu_matrix.cuh"

std::set<FeatType *> CuMatrix::MemoryPool;
CuMatrix::CuMatrix(Matrix M, const cublasHandle_t &handle_)
    : Matrix(M.getRows(), M.getCols(), M.getData()) {
    cudaStat = cudaError_t();
    handle = handle_;
    nnz = 0;
    csrVal = NULL;
    csrColInd = NULL;
    csrRowInd = NULL;
    isSparse = 0;
    deviceMalloc();
    if (getData() != NULL) deviceSetMatrix();
}

void CuMatrix::explicitFree() {
    CuMatrix::MemoryPool.erase(devPtr);
    cudaFree(devPtr);
}

Matrix CuMatrix::getMatrix() {
    updateMatrixFromGPU();
    return Matrix(getRows(), getCols(), getData());
}

void CuMatrix::freeGPU() {
    for (auto ptr : MemoryPool) {
        if (ptr) {
            cudaFree(ptr);
        }
    }
}

void CuMatrix::loadSpCSR(cusparseHandle_t &handle, Graph &graph) {
    unsigned total = graph.dstGhostCnt + graph.localVtxCnt;
    isSparse = true;
    nnz = graph.backwardAdj.nnz;

    cudaStat = cudaMalloc((void **)&csrVal, nnz * sizeof(EdgeType));
    assert(cudaStat == cudaSuccess);
    cudaStat = cudaMalloc((void **)&csrColInd, nnz * sizeof(unsigned));
    assert(cudaStat == cudaSuccess);
    cudaStat = cudaMalloc((void **)&csrRowInd, nnz * sizeof(unsigned));
    assert(cudaStat == cudaSuccess);
    cudaStat = cudaMalloc((void **)&csrRowPtr,
                          (graph.localVtxCnt + 1) * sizeof(unsigned));
    assert(cudaStat == cudaSuccess);
    cudaStat = cudaMemcpy(csrVal, graph.backwardAdj.values,
                          sizeof(EdgeType) * nnz, cudaMemcpyHostToDevice);
    assert(cudaStat == cudaSuccess);
    cudaStat = cudaMemcpy(csrColInd, graph.backwardAdj.columnIdxs,
                          sizeof(unsigned) * nnz, cudaMemcpyHostToDevice);
    assert(cudaStat == cudaSuccess);
    unsigned *rowPtrs = new unsigned[graph.localVtxCnt + 1];
    for (unsigned i = 0; i < graph.localVtxCnt + 1; ++i) {
        rowPtrs[i] = (unsigned)(graph.backwardAdj.rowPtrs[i]);
    }
    cudaStat = cudaMemcpy(csrRowPtr, rowPtrs,
                          sizeof(unsigned) * (graph.localVtxCnt + 1),
                          cudaMemcpyHostToDevice);
    assert(cudaStat == cudaSuccess);

    setRows(graph.localVtxCnt);
    setCols(total);
}

void CuMatrix::loadSpCSC(cusparseHandle_t &handle, Graph &graph) {
    unsigned total = graph.srcGhostCnt + graph.localVtxCnt;
    isSparse = true;
    nnz = graph.forwardAdj.nnz;

    cudaStat = cudaMalloc((void **)&csrVal, nnz * sizeof(EdgeType));
    assert(cudaStat == cudaSuccess);
    cudaStat = cudaMalloc((void **)&csrColInd, nnz * sizeof(unsigned));
    assert(cudaStat == cudaSuccess);
    cudaStat = cudaMalloc((void **)&csrRowInd, nnz * sizeof(unsigned));
    assert(cudaStat == cudaSuccess);
    cudaStat = cudaMalloc((void **)&csrRowPtr,
                          (graph.localVtxCnt + 1) * sizeof(unsigned));
    assert(cudaStat == cudaSuccess);
    cudaStat = cudaMemcpy(csrVal, graph.forwardAdj.values,
                          sizeof(EdgeType) * nnz, cudaMemcpyHostToDevice);
    assert(cudaStat == cudaSuccess);
    cudaStat = cudaMemcpy(csrColInd, graph.forwardAdj.rowIdxs,
                          sizeof(unsigned) * nnz, cudaMemcpyHostToDevice);
    assert(cudaStat == cudaSuccess);
    unsigned *columnPtrs = new unsigned[graph.localVtxCnt + 1];
    for (unsigned i = 0; i < graph.localVtxCnt + 1; ++i) {
        columnPtrs[i] = (unsigned)(graph.forwardAdj.columnPtrs[i]);
    }
    cudaStat = cudaMemcpy(csrRowPtr, columnPtrs,
                          sizeof(unsigned) * (graph.localVtxCnt + 1),
                          cudaMemcpyHostToDevice);
    assert(cudaStat == cudaSuccess);

    setRows(graph.localVtxCnt);
    setCols(total);
}

void CuMatrix::loadSpDense(FeatType *vtcsTensor, FeatType *ghostTensor,
                           unsigned numLocalVertices, unsigned numGhostVertices,
                           unsigned numFeat) {
    // Still row major
    unsigned totalVertices = (numLocalVertices + numGhostVertices);
    cudaStat = cudaMalloc((void **)&devPtr,
                          numFeat * sizeof(FeatType) * totalVertices);
    assert(cudaStat == cudaSuccess);
    cudaStat = cudaMemcpy(devPtr, vtcsTensor,
                          sizeof(FeatType) * numLocalVertices * numFeat,
                          cudaMemcpyHostToDevice);
    assert(cudaStat == cudaSuccess);
    cudaStat = cudaMemcpy(devPtr + numLocalVertices * numFeat, ghostTensor,
                          sizeof(FeatType) * numGhostVertices * numFeat,
                          cudaMemcpyHostToDevice);
    assert(cudaStat == cudaSuccess);
    setRows(totalVertices);
    setCols(numFeat);
    MemoryPool.insert(devPtr);
}

CuMatrix CuMatrix::extractRow(unsigned row) {
    FeatType *data = getData() ? (getData() + row * getCols()) : NULL;
    CuMatrix rowVec;
    rowVec.handle = handle;
    rowVec.setData(data);
    rowVec.setRows(1);
    rowVec.setCols(getCols());
    rowVec.devPtr = devPtr + row * getCols();
    return rowVec;
}

void CuMatrix::deviceMalloc() {
    unsigned rows = this->getRows();
    unsigned cols = this->getCols();
    cudaStat = cudaMalloc((void **)&devPtr, rows * cols * sizeof(FeatType));
    if (cudaStat != cudaSuccess) {
        printf("device memory allocation failed %u\n", cudaStat);
        exit(EXIT_FAILURE);
    }
    MemoryPool.insert(devPtr);
}

void CuMatrix::deviceSetMatrix() {
    unsigned rows = this->getRows();
    unsigned cols = this->getCols();
    FeatType *data = this->getData();

    stat = cublasSetMatrix(rows, cols, sizeof(float), data, rows, devPtr, rows);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        switch (stat) {
            case CUBLAS_STATUS_NOT_INITIALIZED:
                printf("CUBLAS_STATUS_NOT_INITIALIZED\n");
                break;
            case CUBLAS_STATUS_INVALID_VALUE:
                printf("CUBLAS_STATUS_INVALID_VALUE\n");
                break;
            case CUBLAS_STATUS_MAPPING_ERROR:
                printf("CUBLAS_STATUS_MAPPING_ERROR\n");
                break;
        }
        cudaFree(devPtr);
        cublasDestroy(handle);
        exit(EXIT_FAILURE);
    }
}

void CuMatrix::updateMatrixFromGPU() {
    unsigned rows = this->getRows();
    unsigned cols = this->getCols();
    if (getData() == NULL) setData(new FeatType[getNumElemts()]);
    FeatType *data = this->getData();
    stat = cublasGetMatrix(rows, cols, sizeof(float), devPtr, rows, data, rows);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf("data upload failed\n");
        switch (stat) {
            case CUBLAS_STATUS_NOT_INITIALIZED:
                printf("CUBLAS_STATUS_NOT_INITIALIZED\n");
                break;
            case CUBLAS_STATUS_INVALID_VALUE:
                printf("CUBLAS_STATUS_INVALID_VALUE\n");
                break;
            case CUBLAS_STATUS_MAPPING_ERROR:
                printf("CUBLAS_STATUS_MAPPING_ERROR\n");
                break;
        }
        cudaFree(devPtr);
        cublasDestroy(handle);
        exit(EXIT_FAILURE);
    }
}

CuMatrix::~CuMatrix() {}

void CuMatrix::scale(const float &alpha) {
    cublasSscal(handle, getNumElemts(), &alpha, devPtr, 1);
}

CuMatrix CuMatrix::dot(CuMatrix &B, bool A_trans, bool B_trans, float alpha,
                       float beta) {
    if (handle != B.handle) {
        std::cout << "Handle don't match\n";
        exit(EXIT_FAILURE);
    }
    cublasOperation_t ATrans = A_trans ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t BTrans = B_trans ? CUBLAS_OP_T : CUBLAS_OP_N;
    // 1. cublas is using col-major
    // 2. when cpy into/out device memory, it will do Transpose
    // 3. C=AB and C^T= (B^T*A^T)
    // This means just swap the order of multiplicaiton
    // Guide: https://peterwittek.com/cublas-matrix-c-style.html
    Matrix AT = Matrix(getCols(), getRows(), getData());
    Matrix BT = Matrix(B.getCols(), B.getRows(), B.getData());

    unsigned CRow = A_trans ? AT.getRows() : getRows();
    unsigned CCol = B_trans ? BT.getCols() : B.getCols();
    Matrix mat_C(CRow, CCol, (char *)NULL);  // real C

    unsigned k = A_trans ? getRows() : getCols();
    CuMatrix C(mat_C, handle);

    stat = cublasSgemm(handle, BTrans, ATrans, C.getCols(), C.getRows(), k,
                       &alpha, B.devPtr, B.getCols(), devPtr, getCols(), &beta,
                       C.devPtr, C.getCols());
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf("SGEMM ERROR\n");
        cudaFree(devPtr);
        cublasDestroy(handle);
        exit(EXIT_FAILURE);
    }
    return C;
}

CuMatrix CuMatrix::transpose() {
    CuMatrix res(Matrix(getCols(), getRows(), (FeatType *)NULL), handle);
    float alpha = 1.0;
    float beta = 0.;

    stat = cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, getRows(), getCols(),
                       &alpha, devPtr, getCols(), &beta, devPtr, getRows(),
                       res.devPtr, getRows());
    if (stat != CUBLAS_STATUS_SUCCESS) {
        cublasDestroy(handle);
        exit(EXIT_FAILURE);
    }

    return res;
}
