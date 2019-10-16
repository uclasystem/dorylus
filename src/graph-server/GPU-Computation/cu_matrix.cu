#include "cu_matrix.cuh"

CuMatrix::CuMatrix( Matrix M, const cublasHandle_t & handle_)
    :Matrix(M.getRows(),M.getCols(),M.getData())
{   
    cudaStat=cudaError_t();
    handle=handle_;
    deviceMalloc();
    if(getData()!=NULL)
       deviceSetMatrix();
}

Matrix CuMatrix::getMatrix(){
    updateMatrixFromGPU();
    return Matrix(getRows(),getCols(),getData());
}

void CuMatrix::deviceMalloc(){
    unsigned rows=this->getRows();
    unsigned cols=this->getCols();
   
    cudaStat = cudaMalloc ((void**)&devPtr, rows*cols*sizeof(FeatType));
    if (cudaStat != cudaSuccess) {
        printf("%u\n", cudaStat);
        printf ("device memory allocation failed\n");
        exit (EXIT_FAILURE);
    }
}
void CuMatrix::deviceSetMatrix(){
    unsigned rows=this->getRows();
    unsigned cols=this->getCols();
    FeatType * data=this->getData();
    stat = cublasSetMatrix (rows,cols, sizeof(float), data, rows , devPtr, rows);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        switch (stat){
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
        
        printf ("data download failed\n");
        cudaFree (devPtr);
        cublasDestroy(handle);
        exit (EXIT_FAILURE);
    }
}


void CuMatrix::updateMatrixFromGPU(){
    unsigned rows=this->getRows();
    unsigned cols=this->getCols();
    if(getData()==NULL)
        setData(new FeatType[getNumElemts()]);
    FeatType * data=this->getData();
    stat = cublasGetMatrix (rows, cols, sizeof(float), devPtr, rows, data, rows );
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("data upload failed\n");
        cudaFree (devPtr);
        cublasDestroy(handle);
        exit (EXIT_FAILURE);
    }
}

CuMatrix::~CuMatrix(){
    cudaFree (devPtr);
}

CuMatrix CuMatrix::dot(CuMatrix& M,float alpha,float beta){
    if(handle!=M.handle){
        std::cout<<"Handle don't match\n";
        exit(EXIT_FAILURE);
    }
    Matrix mat_C(getRows(),M.getCols(),new FeatType[getRows()*M.getCols()]);
    CuMatrix C(mat_C,handle);
    //1. cublas is using col-major
    //2. when cpy into/out device memory, it will do Transpose 
    //3. C=AB and C^T= (B^T*A^T)
    //This means just swap the order of multiplicaiton
    //Guide: https://peterwittek.com/cublas-matrix-c-style.html
    cublasSgemm(handle,
        CUBLAS_OP_N,CUBLAS_OP_N,
        M.getCols(),getRows(),M.getRows(),
        &alpha,
        M.devPtr,M.getCols(),
        devPtr,getCols(),
        &beta,
        C.devPtr,M.getCols());
    return C;
}

CuMatrix CuMatrix::transpose(){
    float alpha=1.0;
    float beta=0.;
    CuMatrix res(Matrix(getCols(),getRows(),(FeatType*) NULL),handle);
    cublasSgeam(handle,CUBLAS_OP_T,CUBLAS_OP_T,
        getRows(),getCols(),
        &alpha,
        devPtr,getCols(),
        &beta,
        devPtr,getCols(),
        res.devPtr,getRows());
    return res;
}