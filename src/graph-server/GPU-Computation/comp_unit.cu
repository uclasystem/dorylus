#include "comp_unit.cuh"
#include "cuda_ops.cuh"


ComputingUnit::ComputingUnit(){
    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS initialization failed\n");
        printf ("CUBLAS stat %u\n",stat);
        exit (EXIT_FAILURE);
    }
    cublasSetMathMode(handle,CUBLAS_TENSOR_OP_MATH);
}

CuMatrix ComputingUnit::wrapMatrix(Matrix m){
    return CuMatrix(m,handle);
}


CuMatrix ComputingUnit::hadamardSub(CuMatrix& matLeft,CuMatrix& matRight) {
    assert(matLeft.getRows() == matRight.getRows());
    assert(matLeft.getCols() == matRight.getCols());
    // CuMatrix res(Matrix(matLeft.getRows(),matLeft.getCols(), new FeatType[matLeft.getNumElemts()]),handle);
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

// GPU is faster than CPU when SIZE>~250000, Maybe CPU is faster in most cases 
CuMatrix ComputingUnit::softmaxRows( CuMatrix &mat){
    CuMatrix res(Matrix(mat.getRows(),mat.getCols(),(FeatType *)NULL),handle);
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

CuMatrix ComputingUnit::activateDerivative( CuMatrix& mat) {
    CuMatrix res(Matrix(mat.getRows(),mat.getCols(),(FeatType *)NULL),handle);
    thrust::device_ptr<float> res_ptr(res.devPtr);
    CuMatrix z(Matrix(mat.getRows(),mat.getCols(),mat.getData()),handle);
    thrust::device_ptr<float> z_ptr(z.devPtr);
    thrust::transform(z_ptr, z_ptr+mat.getNumElemts(),res_ptr, activateDerivative_functor());
    return res;
}


CuMatrix
ComputingUnit::dotGDwithWTrans( CuMatrix& matLeft, CuMatrix& matRight) {
    CuMatrix matRightTrans = matRight.transpose();
    CuMatrix res = matLeft.dot(matRightTrans);
    return res;
}

CuMatrix
ComputingUnit::dotActTranswithGD( CuMatrix& matLeft, CuMatrix& matRight, float learning_rate) {
    CuMatrix matLeftTrans = matLeft.transpose();
    CuMatrix res=matLeftTrans.dot(matRight,learning_rate);
    return res;
}



CuMatrix ComputingUnit::dot( Matrix& A, Matrix& B){
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
