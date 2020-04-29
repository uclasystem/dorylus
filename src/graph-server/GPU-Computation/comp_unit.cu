#include "comp_unit.cuh"
#include "cuda_ops.cuh"

#define ACTIVATION CUDNN_ACTIVATION_TANH

const float alpha = 1.0f, beta = 0.0f;
using namespace std;
void cudaErrCheck(cudaError_t stat) {
    if (stat != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(stat));
        fprintf(stderr, "CUDA Error: %d\n", (stat));
    }
}

ComputingUnit *ComputingUnit::instance = nullptr;
ComputingUnit &ComputingUnit::getInstance() {
    if (instance == nullptr) instance = new ComputingUnit();
    return *instance;
}

ComputingUnit::ComputingUnit() {
    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf("CUBLAS initialization failed\n");
        printf("CUBLAS stat %u\n", stat);
        exit(EXIT_FAILURE);
    }
    cudnnStatus_t err = cudnnCreate(&cudnnHandle);
    if (err != CUDNN_STATUS_SUCCESS) {
        std::cout << "Error occurred: " << err << std::endl;
        std::exit(EXIT_FAILURE);
    }
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
    auto cusparseStat = cusparseCreate(&spHandle);
    assert(CUSPARSE_STATUS_SUCCESS == cusparseStat);
}

CuMatrix ComputingUnit::wrapMatrix(Matrix m) { return CuMatrix(m, handle); }

CuMatrix ComputingUnit::aggregate(CuMatrix &sparse, CuMatrix &dense,
                                  CuMatrix &norms) {
    CuMatrix C(Matrix(dense.getCols(), sparse.getRows(), (FeatType *)NULL),
               handle);

    cusparseSpMatDescr_t desA;
    cusparseDnMatDescr_t desB;
    cusparseDnMatDescr_t desC;

    const float agg_beta = 0;
    auto cusparseStat = cusparseCreateCsr(
        &desA, sparse.getRows(), sparse.getCols(), sparse.nnz, sparse.csrRowPtr,
        sparse.csrColInd, sparse.csrVal, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    assert(CUSPARSE_STATUS_SUCCESS == cusparseStat);
    cusparseStat = cusparseCreateDnMat(&desB, dense.getCols(), dense.getRows(),
                                       dense.getCols(), dense.devPtr,
                                       CUDA_R_32F, CUSPARSE_ORDER_COL);
    assert(CUSPARSE_STATUS_SUCCESS == cusparseStat);
    cusparseStat = cusparseCreateDnMat(&desC, sparse.getRows(), dense.getCols(),
                                       sparse.getRows(), C.devPtr, CUDA_R_32F,
                                       CUSPARSE_ORDER_COL);
    assert(CUSPARSE_STATUS_SUCCESS == cusparseStat);

    std::size_t buffer_size;
    cusparseStat = cusparseSpMM_bufferSize(
        spHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_TRANSPOSE, &alpha, desA, desB, &agg_beta, desC,
        CUDA_R_32F, CUSPARSE_MM_ALG_DEFAULT, &buffer_size);
    assert(CUSPARSE_STATUS_SUCCESS == cusparseStat);
    float *buffer;
    cudaErrCheck(cudaMalloc((void **)&buffer, buffer_size * sizeof(float)));
    cusparseStat = cusparseSpMM(spHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                CUSPARSE_OPERATION_TRANSPOSE, &alpha, desA,
                                desB, &agg_beta, desC, CUDA_R_32F,
                                CUSPARSE_MM_ALG_DEFAULT, buffer);
    assert(CUSPARSE_STATUS_SUCCESS == cusparseStat);
    C = C.transpose();
    scaleRowsByVector(dense, norms);

    hadamardAdd(C, dense);
    cudaDeviceSynchronize();
    return C;
}
// This function will scale first nth rows of M based on the length of cuV
void ComputingUnit::scaleRowsByVector(CuMatrix &cuM, CuMatrix &cuV) {
    stat = cublasSdgmm(handle, CUBLAS_SIDE_RIGHT, cuM.getCols(), cuV.getRows(),
                       cuM.devPtr, cuM.getCols(), cuV.devPtr, 1, cuM.devPtr,
                       cuM.getCols());
    assert(stat == CUBLAS_STATUS_SUCCESS);
}

void ComputingUnit::hadamardAdd(CuMatrix &matLeft, CuMatrix &matRight) {
    thrust::device_ptr<float> cuLeft_ptr(matLeft.devPtr);
    thrust::device_ptr<float> cuRight_ptr(matRight.devPtr);
    thrust::transform(cuLeft_ptr, cuLeft_ptr + matLeft.getNumElemts(),
                      cuRight_ptr, cuLeft_ptr, thrust::plus<float>());
}

CuMatrix ComputingUnit::hadamardSub(CuMatrix &matLeft, CuMatrix &matRight) {
    assert(matLeft.getRows() == matRight.getRows());
    assert(matLeft.getCols() == matRight.getCols());
    CuMatrix res(Matrix(matLeft.getRows(), matLeft.getCols(), (FeatType *)NULL),
                 handle);

    thrust::device_ptr<float> cuLeft_ptr(matLeft.devPtr);
    thrust::device_ptr<float> cuRight_ptr(matRight.devPtr);
    thrust::device_ptr<float> res_ptr(res.devPtr);
    thrust::transform(cuLeft_ptr, cuLeft_ptr + matLeft.getNumElemts(),
                      cuRight_ptr, res_ptr, thrust::minus<float>());
    return res;
}

CuMatrix ComputingUnit::hadamardMul(CuMatrix &matLeft, CuMatrix &matRight) {
    assert(matLeft.getRows() == matRight.getRows());
    assert(matLeft.getCols() == matRight.getCols());
    CuMatrix res(Matrix(matLeft.getRows(), matLeft.getCols(), (FeatType *)NULL),
                 handle);

    thrust::device_ptr<float> cuLeft_ptr(matLeft.devPtr);
    thrust::device_ptr<float> cuRight_ptr(matRight.devPtr);
    thrust::device_ptr<float> res_ptr(res.devPtr);

    thrust::transform(cuLeft_ptr, cuLeft_ptr + matLeft.getNumElemts(),
                      cuRight_ptr, res_ptr, thrust::multiplies<float>());

    return res;
}

CuMatrix ComputingUnit::softmaxRows(CuMatrix &mat) {
    CuMatrix res(Matrix(mat.getRows(), mat.getCols(), (FeatType *)NULL),
                 handle);
    cudnnTensorDescriptor_t srcTensorDesc, sftTensorDesc;
    cudnnCreateTensorDescriptor(&srcTensorDesc);
    cudnnCreateTensorDescriptor(&sftTensorDesc);
    cudnnSetTensor4dDescriptor(srcTensorDesc, CUDNN_TENSOR_NCHW,
                               CUDNN_DATA_FLOAT, mat.getRows(), 1, 1,
                               mat.getCols());
    cudnnSetTensor4dDescriptor(sftTensorDesc, CUDNN_TENSOR_NCHW,
                               CUDNN_DATA_FLOAT, mat.getRows(), 1, 1,
                               mat.getCols());
    cudnnSoftmaxForward(cudnnHandle, CUDNN_SOFTMAX_ACCURATE,
                        CUDNN_SOFTMAX_MODE_INSTANCE, &alpha, srcTensorDesc,
                        mat.devPtr, &beta, sftTensorDesc, res.devPtr);
    return res;
}

CuMatrix ComputingUnit::activateBackward(CuMatrix &x, CuMatrix &y,
                                         CuMatrix &dy) {
    FeatType *x_d = new FeatType[y.getNumElemts()];
    FeatType *dx_d = new FeatType[y.getNumElemts()];
    memset(dx_d, 0, y.getDataSize());
    memset(x_d, 0, y.getDataSize());
    CuMatrix dx(Matrix(y.getRows(), y.getCols(), dx_d), handle);
    CuMatrix x_(Matrix(y.getRows(), y.getCols(), x_d), handle);

    cudnnActivationDescriptor_t actDesc;
    cudnnCreateActivationDescriptor(&actDesc);
    cudnnSetActivationDescriptor(actDesc, ACTIVATION, CUDNN_NOT_PROPAGATE_NAN,
                                 0.0);
    cudnnTensorDescriptor_t yDesc;
    cudnnCreateTensorDescriptor(&yDesc);
    cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                               y.getRows(), 1, 1, y.getCols());
    auto error = cudnnActivationBackward(cudnnHandle, actDesc, &alpha, yDesc,
                                         y.devPtr, yDesc, dy.devPtr, yDesc,
                                         x_.devPtr, &beta, yDesc, dx.devPtr);
    assert(CUDNN_STATUS_SUCCESS == error);
    return dx;
}

CuMatrix ComputingUnit::dot(Matrix &A, Matrix &B) {
    CuMatrix devA(A, handle);
    CuMatrix devB(B, handle);
    CuMatrix devC = devA.dot(devB);
    devC.updateMatrixFromGPU();
    return devC;
}

void ComputingUnit::activate(CuMatrix &A) {
    cudnnTensorDescriptor_t srcTensorDesc;
    cudnnCreateTensorDescriptor(&srcTensorDesc);
    cudnnSetTensor4dDescriptor(srcTensorDesc, CUDNN_TENSOR_NCHW,
                               CUDNN_DATA_FLOAT, A.getRows(), 1, 1,
                               A.getCols());

    cudnnActivationDescriptor_t actDesc;
    cudnnCreateActivationDescriptor(&actDesc);
    cudnnSetActivationDescriptor(actDesc, ACTIVATION, CUDNN_PROPAGATE_NAN, 0.0);
    cudnnActivationForward(cudnnHandle, actDesc, &alpha, srcTensorDesc,
                           A.devPtr, &beta, srcTensorDesc, A.devPtr);
}

//** much slower than CPU only if Input Matrices are not loaded in GPU
// beforehand
unsigned ComputingUnit::checkAccuracy(CuMatrix &predictions, CuMatrix &labels) {
    unsigned rowSize = predictions.getCols();

    thrust::device_vector<FeatType *> row_starts(predictions.getRows());
    thrust::counting_iterator<int> idxfirst(0);

    thrust::transform(idxfirst, idxfirst + predictions.getRows(),
                      row_starts.begin(),
                      setRowStarts(predictions.devPtr, rowSize));
    thrust::device_vector<unsigned> pred_results(predictions.getRows());
    thrust::transform(row_starts.begin(), row_starts.end(),
                      pred_results.begin(), findRowMaximum(rowSize));

    thrust::transform(idxfirst, idxfirst + predictions.getRows(),
                      row_starts.begin(), setRowStarts(labels.devPtr, rowSize));
    thrust::device_vector<unsigned> true_results(predictions.getRows());
    thrust::transform(pred_results.begin(), pred_results.end(),
                      row_starts.begin(), true_results.begin(),
                      isPredictCorrect(rowSize));

    unsigned totalCorrect =
        thrust::reduce(true_results.begin(), true_results.end(), (unsigned)0,
                       thrust::plus<unsigned>());
    return totalCorrect;
}

//** much slower than CPU only if Input Matrices are not loaded in GPU
// beforehand
float ComputingUnit::checkLoss(CuMatrix &preds, CuMatrix &labels) {
    unsigned rowSize = preds.getCols();

    thrust::counting_iterator<int> idxfirst(0);
    thrust::device_vector<FeatType *> row_starts(preds.getRows());
    thrust::transform(idxfirst, idxfirst + preds.getRows(), row_starts.begin(),
                      setRowStarts(labels.devPtr, rowSize));
    thrust::device_vector<unsigned> true_labels(preds.getRows());
    thrust::transform(row_starts.begin(), row_starts.end(), true_labels.begin(),
                      findTrueLabel(rowSize));
    thrust::transform(idxfirst, idxfirst + preds.getRows(), row_starts.begin(),
                      setRowStarts(preds.devPtr, rowSize));
    thrust::device_vector<FeatType> losses(preds.getRows());
    thrust::transform(true_labels.begin(), true_labels.end(),
                      row_starts.begin(), losses.begin(), getLoss(rowSize));
    float totalLoss = thrust::reduce(losses.begin(), losses.end(), (float)0,
                                     thrust::plus<float>());
    return totalLoss;
}

void ComputingUnit::getTrainStat(CuMatrix &preds, CuMatrix &labels, float &acc,
                                 float &loss) {
    loss = checkLoss(preds, labels) / labels.getRows();
    acc = checkAccuracy(preds, labels) / (float)labels.getRows();
    // float * l = new float [labels.getNumElemts()];
    // float * p = new float [preds.getNumElemts()];
    // preds.setData(p);
    // labels.setData(l);
    // preds.updateMatrixFromGPU();
    // labels.updateMatrixFromGPU();
    // acc = 0.0;
    // loss = 0.0;
    // unsigned featDim=labels.getCols();
    // for (unsigned i = 0; i < labels.getRows(); i++) {
    //     FeatType *currLabel = l + i * labels.getCols();
    //     FeatType *currPred = p + i * labels.getCols();
    //     acc += currLabel[argmax(currPred, currPred + featDim)];
    //     loss -= std::log(currPred[argmax(currLabel, currLabel + featDim)]);
    // }
    // acc /= labels.getRows();
    // loss /= labels.getRows();
    // printLog(getNodeId(), "batch loss %f, batch acc %f", loss, acc);
}