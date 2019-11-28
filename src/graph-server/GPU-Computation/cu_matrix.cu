#include "cu_matrix.cuh"
#include <tuple>


typedef std::tuple<int, int, EdgeType> triplet;

std::set<FeatType *> CuMatrix::MemoryPool;
CuMatrix::CuMatrix( Matrix M, const cublasHandle_t &handle_)
    : Matrix(M.getRows(), M.getCols(), M.getData()) {
    cudaStat = cudaError_t();
    handle = handle_;
    nnz = 0;
    csrVal = NULL;
    csrRowPtr = NULL;
    csrColInd = NULL;
    isSparse = 0;
    deviceMalloc();
    if(getData() != NULL)
        deviceSetMatrix();
}

Matrix CuMatrix::getMatrix() {
    updateMatrixFromGPU();
    return Matrix(getRows(), getCols(), getData());
}

void CuMatrix::freeGPU() {
    for(auto ptr : MemoryPool)
        cudaFree (ptr);
}

void CuMatrix::loadSpCSR(cusparseHandle_t &handle, unsigned numLocalVertices, std::vector<Vertex> &vertices, unsigned numGhostVertices) {
    unsigned total = numGhostVertices + numLocalVertices;
    isSparse = true;

    //GET COO FORMAT FIRST
    unsigned count = 0;
    std::set<triplet> links;
    for(auto &v : vertices) {
        links.insert(triplet(v.getLocalId(), v.getLocalId(), v.getNormFactor()));
        count++;
        for(unsigned i = 0; i < v.getNumInEdges(); ++i) {
            InEdge &ie = v.getInEdge(i);
            if(ie.getEdgeLocation() == LOCAL_EDGE_TYPE)
                links.insert(triplet(ie.getSourceId(), v.getLocalId(), ie.getData()));
            else
                links.insert(triplet(ie.getSourceId() + numLocalVertices, v.getLocalId(), ie.getData()));
            count++;
        }
    }
    nnz = count;
    EdgeType *norms  = new EdgeType[count];
    int *rowInd = new int[count];
    int *colInd = new int[count];

    unsigned i = 0;
    for(auto link : links) {
        rowInd[i] = std::get<0>(link);
        colInd[i] = std::get<1>(link);
        norms[i] = std::get<2>(link);
        i++;
    }

    int *cooRowInd;
    //coo device pointer
    cudaStat = cudaMalloc ((void **)&csrVal, count * sizeof(EdgeType));
    assert(cudaStat == cudaSuccess);
    cudaStat = cudaMalloc ((void **)&cooRowInd, count * sizeof(int));
    assert(cudaStat == cudaSuccess);
    cudaStat = cudaMalloc ((void **)&csrColInd, count * sizeof(int));
    assert(cudaStat == cudaSuccess);
    cudaStat = cudaMalloc ((void **)&csrRowPtr, (total + 1) * sizeof(int));
    assert(cudaStat == cudaSuccess);
    cudaStat = cudaMemcpy(csrVal, norms, sizeof(EdgeType) * count, cudaMemcpyHostToDevice);
    assert(cudaStat == cudaSuccess);
    cudaStat = cudaMemcpy(cooRowInd, rowInd, sizeof(int) * count, cudaMemcpyHostToDevice);
    assert(cudaStat == cudaSuccess);
    cudaStat = cudaMemcpy(csrColInd, colInd, sizeof(int) * count, cudaMemcpyHostToDevice);
    assert(cudaStat == cudaSuccess);
    //convert to CSR in GPU
    auto status = cusparseXcoo2csr(handle, cooRowInd, nnz, total, csrRowPtr, CUSPARSE_INDEX_BASE_ZERO);
    assert(status == CUSPARSE_STATUS_SUCCESS);
    setRows(total);
    setCols(total);

    cudaFree(cooRowInd);
    delete[] rowInd;
    delete[] colInd;
    delete[] norms;
}

void CuMatrix::loadSpDense(FeatType *vtcsTensor, FeatType *ghostTensor,
                           unsigned numLocalVertices, unsigned numGhostVertices,
                           unsigned numFeat) {
    //Still row major
    unsigned totalVertices = (numLocalVertices + numGhostVertices);
    cudaStat = cudaMalloc ((void **)&devPtr,  numFeat * sizeof(FeatType) * totalVertices);
    cudaMemcpy(devPtr, vtcsTensor, sizeof(FeatType)*numLocalVertices * numFeat, cudaMemcpyHostToDevice);
    cudaMemcpy(devPtr  + numLocalVertices * numFeat, ghostTensor,
               sizeof(FeatType)*numGhostVertices * numFeat, cudaMemcpyHostToDevice);
    setRows(totalVertices);
    setCols(numFeat);
    MemoryPool.insert(devPtr);
    cudaDeviceSynchronize();
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

    cudaStat = cudaMalloc ((void **)&devPtr, rows * cols * sizeof(FeatType));
    MemoryPool.insert(devPtr);

    if (cudaStat != cudaSuccess) {
        printf("%u\n", cudaStat);
        printf ("device memory allocation failed\n");
        exit (EXIT_FAILURE);
    }
}

void CuMatrix::deviceSetMatrix() {
    unsigned rows = this->getRows();
    unsigned cols = this->getCols();
    FeatType *data = this->getData();
    stat = cublasSetMatrix (rows, cols, sizeof(float), data, rows, devPtr, rows);
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
        cudaFree (devPtr);
        cublasDestroy(handle);
        exit (EXIT_FAILURE);
    }
}


void CuMatrix::updateMatrixFromGPU() {
    unsigned rows = this->getRows();
    unsigned cols = this->getCols();
    if(getData() == NULL)
        setData(new FeatType[getNumElemts()]);
    FeatType *data = this->getData();
    stat = cublasGetMatrix (rows, cols, sizeof(float), devPtr, rows, data, rows );
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("data upload failed\n");
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
        cudaFree (devPtr);
        cublasDestroy(handle);
        exit (EXIT_FAILURE);
    }
}

CuMatrix::~CuMatrix() {}

void CuMatrix::scale(const float &alpha) {
    cublasSscal(handle, getNumElemts(), &alpha, devPtr, 1);
}

CuMatrix CuMatrix::dot(CuMatrix &B, bool A_trans, bool B_trans, float alpha, float beta) {
    if(handle != B.handle) {
        std::cout << "Handle don't match\n";
        exit(EXIT_FAILURE);
    }
    cublasOperation_t ATrans = A_trans ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t BTrans = B_trans ? CUBLAS_OP_T : CUBLAS_OP_N;
    //1. cublas is using col-major
    //2. when cpy into/out device memory, it will do Transpose
    //3. C=AB and C^T= (B^T*A^T)
    //This means just swap the order of multiplicaiton
    //Guide: https://peterwittek.com/cublas-matrix-c-style.html
    Matrix AT = Matrix(getCols(), getRows(), getData());
    Matrix BT = Matrix(B.getCols(), B.getRows(), B.getData());

    unsigned CRow = A_trans ? AT.getRows() : getRows();
    unsigned CCol = B_trans ? BT.getCols() : B.getCols();
    Matrix mat_C(CRow, CCol, (char *)NULL); //real C

    unsigned k = A_trans ? getRows() : getCols();
    CuMatrix C(mat_C, handle);

    stat = cublasSgemm(handle,
                       BTrans, ATrans,
                       C.getCols(), C.getRows(), k,
                       &alpha,
                       B.devPtr, B.getCols(),
                       devPtr, getCols(),
                       &beta,
                       C.devPtr, C.getCols());
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("SGEMM ERROR\n");
        cudaFree (devPtr);
        cublasDestroy(handle);
        exit (EXIT_FAILURE);
    }
    return C;
}

CuMatrix CuMatrix::transpose() {
    float alpha = 1.0;
    float beta = 0.;
    CuMatrix res(Matrix(getCols(), getRows(), (FeatType *) NULL), handle);
    cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T,
                getRows(), getCols(),
                &alpha,
                devPtr, getCols(),
                &beta,
                devPtr, getCols(),
                res.devPtr, getRows());
    return res;
}
