#include "comp_unit.cuh"
#include "cuda_ops.cuh"
const float alpha=1.0,beta=0.0;
  
ComputingUnit::ComputingUnit(){
    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS initialization failed\n");
        printf ("CUBLAS stat %u\n",stat);
        exit (EXIT_FAILURE);
    }
    cudnnStatus_t err =cudnnCreate(&cudnnHandle); 
    if (err != CUDNN_STATUS_SUCCESS) { 
        std::cout << "Error occurred: " << err << std::endl; 
        std::exit(EXIT_FAILURE); 
    }
    cublasSetMathMode(handle,CUBLAS_TENSOR_OP_MATH);
}

CuMatrix ComputingUnit::wrapMatrix(Matrix m){
    return CuMatrix(m,handle);
}


CuMatrix ComputingUnit::hadamardSub(CuMatrix& matLeft,CuMatrix& matRight) {
    assert(matLeft.getRows() == matRight.getRows());
    assert(matLeft.getCols() == matRight.getCols());
    CuMatrix res(Matrix(matLeft.getRows(),matLeft.getCols(), (FeatType *)NULL),handle);

    thrust::device_ptr<float> cuLeft_ptr(matLeft.devPtr);
    thrust::device_ptr<float> cuRight_ptr(matRight.devPtr);
    thrust::device_ptr<float> res_ptr(res.devPtr);

    thrust::transform(cuLeft_ptr,cuLeft_ptr+matLeft.getNumElemts(),
                        cuRight_ptr,
                        res_ptr,
                        thrust::minus<float>());

    return res;
}

CuMatrix ComputingUnit::hadamardMul( CuMatrix& matLeft, CuMatrix& matRight) {
    assert(matLeft.getRows() == matRight.getRows());
    assert(matLeft.getCols() == matRight.getCols());
    CuMatrix res(Matrix(matLeft.getRows(),matLeft.getCols(), (FeatType *)NULL),handle);

    thrust::device_ptr<float> cuLeft_ptr(matLeft.devPtr);
    thrust::device_ptr<float> cuRight_ptr(matRight.devPtr);
    thrust::device_ptr<float> res_ptr(res.devPtr);

    thrust::transform(cuLeft_ptr,cuLeft_ptr+matLeft.getNumElemts(),
                        cuRight_ptr,
                        res_ptr,
                        thrust::multiplies<float>());

    return res;
}

CuMatrix ComputingUnit::softmaxRows( CuMatrix &mat){
    CuMatrix res(Matrix(mat.getRows(),mat.getCols(),(FeatType *)NULL),handle);
    cudnnTensorDescriptor_t srcTensorDesc, sftTensorDesc;
    cudnnCreateTensorDescriptor(&srcTensorDesc);
    cudnnCreateTensorDescriptor(&sftTensorDesc);
    cudnnSetTensor4dDescriptor(srcTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
            mat.getRows(),1,1,mat.getCols());
    cudnnSetTensor4dDescriptor(sftTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
            mat.getRows(),1,1,mat.getCols());
    cudnnSoftmaxForward(cudnnHandle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE,
            &alpha,srcTensorDesc, mat.devPtr, 
            &beta, sftTensorDesc, res.devPtr);
    return res;
}

CuMatrix ComputingUnit::activateBackward( CuMatrix& y,CuMatrix& gradient){
    cudnnActivationDescriptor_t actDesc;
    cudnnCreateActivationDescriptor(&actDesc);
    cudnnSetActivationDescriptor(actDesc,CUDNN_ACTIVATION_RELU,CUDNN_NOT_PROPAGATE_NAN,1.0);

    cudnnTensorDescriptor_t yDesc,dyDesc;
    cudnnCreateTensorDescriptor(&yDesc);
    cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
            y.getRows(),1,1,y.getCols());
    cudnnCreateTensorDescriptor(&dyDesc);
    cudnnSetTensor4dDescriptor(dyDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
            gradient.getRows(),1,1,gradient.getCols());

    cudnnActivationBackward(cudnnHandle,actDesc,
        &alpha, yDesc,y.devPtr,
        dyDesc, gradient.devPtr,
        yDesc, y.devPtr,
        &beta, dyDesc, gradient.devPtr
    );
    return gradient;
}


CuMatrix ComputingUnit::dot( Matrix& A, Matrix& B){
    CuMatrix devA(A,handle);
    CuMatrix devB(B,handle);
    CuMatrix devC=devA.dot(devB);
    devC.updateMatrixFromGPU();
    return devC;
}

void ComputingUnit::activate(CuMatrix& A){
    cudnnTensorDescriptor_t srcTensorDesc;
    cudnnCreateTensorDescriptor(&srcTensorDesc);
    cudnnSetTensor4dDescriptor(srcTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
            A.getRows(),1,1,A.getCols());

    cudnnActivationDescriptor_t actDesc;
    cudnnCreateActivationDescriptor(&actDesc);
    cudnnSetActivationDescriptor(actDesc,CUDNN_ACTIVATION_RELU,CUDNN_NOT_PROPAGATE_NAN,1.0);
    cudnnActivationForward(cudnnHandle,actDesc,
        &alpha,srcTensorDesc,A.devPtr,&beta,srcTensorDesc,A.devPtr);
    A.updateMatrixFromGPU();
}


//** much slower than CPU only if Input Matrices are not loaded in GPU beforehand
unsigned ComputingUnit::checkAccuracy(CuMatrix& predictions, CuMatrix& labels){
    
    unsigned rowSize = predictions.getCols();

    thrust::device_vector<FeatType*> row_starts(predictions.getRows());
    thrust::counting_iterator<int> idxfirst(0);

    thrust::transform(idxfirst,idxfirst+predictions.getRows(),row_starts.begin(),
        setRowStarts(predictions.devPtr,rowSize));
    thrust::device_vector<unsigned> pred_results(predictions.getRows());
    thrust::transform(row_starts.begin(),row_starts.end(),pred_results.begin(),
        findRowMaximum(rowSize));

    thrust::transform(idxfirst,idxfirst+predictions.getRows(),row_starts.begin(),
        setRowStarts(labels.devPtr,rowSize));
    thrust::device_vector<unsigned> true_results(predictions.getRows());
    thrust::transform(pred_results.begin(),pred_results.end(),row_starts.begin(),true_results.begin(),isPredictCorrect(rowSize));

    unsigned totalCorrect = thrust::reduce(true_results.begin(), true_results.end(), (unsigned) 0, thrust::plus<unsigned>());
    return totalCorrect;
}

//** much slower than CPU only if Input Matrices are not loaded in GPU beforehand
float ComputingUnit::checkLoss(CuMatrix& preds, CuMatrix& labels){
    unsigned rowSize=preds.getCols();
    
    thrust::counting_iterator<int> idxfirst(0);
    thrust::device_vector<FeatType*> row_starts(preds.getRows());
    thrust::transform(idxfirst,idxfirst+preds.getRows(),row_starts.begin(),
        setRowStarts(labels.devPtr,rowSize));
    thrust::device_vector<unsigned> true_labels(preds.getRows());
    thrust::transform(row_starts.begin(),row_starts.end(),true_labels.begin(),findTrueLabel(rowSize));
    thrust::transform(idxfirst,idxfirst+preds.getRows(),row_starts.begin(),
        setRowStarts(preds.devPtr,rowSize));
    thrust::device_vector<FeatType> losses(preds.getRows());
    thrust::transform(true_labels.begin(),true_labels.end(),row_starts.begin(),losses.begin(),getLoss(rowSize));
    float totalLoss = thrust::reduce(losses.begin(), losses.end(), (float) 0, thrust::plus<float>());
    return totalLoss;
}
