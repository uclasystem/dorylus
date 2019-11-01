#include "cu_matrix.cuh"
#include "comp_unit.cuh"
#include <iostream>
#include <cudnn.h>
#include <time.h>

using namespace std;
cublasHandle_t handle;
ComputingUnit cu;

const int ROW=5000;
const int COL=41;
float alpha=1.0,beta=0.0;

void test0(){
	float a[6]={1,2,3,4,5,6};
	float b[12]={1,2,3,4,5,6,7,8,9,10,11,12};
	Matrix matA(2,3,a);
	Matrix matB(3,4,b);
	CuMatrix cuA(matA,handle);
	CuMatrix cuB(matB,handle);
	CuMatrix cuD=cuA.dot(cuB,false,false);
	cuD.updateMatrixFromGPU();
	cout<<cuD.str();
	cout<<matA.dot(matB).str();
}
void test1(){
	float a[6]={1,2,3,4,5,6};
	float b[12]={1,2,3,4,5,6,7,8,9,10,11,12};
	Matrix matA(3,2,a);
	Matrix matB(3,4,b);
	CuMatrix cuA(matA,handle);
	CuMatrix cuB(matB,handle);
	CuMatrix cuD=cuA.dot(cuB,true,false);
	cuD.updateMatrixFromGPU();
	cout<<cuD.str();
	cout<<matA.dot(matB,true,false).str();
}
void test2(){
	float a[6]={1,2,3,4,5,6};
	float b[12]={1,2,3,4,5,6,7,8,9,10,11,12};
	Matrix matA(2,3,a);
	Matrix matB(4,3,b);
	CuMatrix cuA(matA,handle);
	CuMatrix cuB(matB,handle);
	CuMatrix cuD=cuA.dot(cuB,false,true);
	cuD.updateMatrixFromGPU();
	cout<<cuD.str();
	cout<<matA.dot(matB,false,true).str();
}
void test3(){
	float a[6]={1,2,3,4,5,6};
	float b[12]={1,2,3,4,5,6,7,8,9,10,11,12};
	Matrix matA(3,2,a);
	Matrix matB(4,3,b);	
	CuMatrix cuA(matA,handle);
	CuMatrix cuB(matB,handle);
	CuMatrix cuD=cuA.dot(cuB,true,true);
	cuD.updateMatrixFromGPU();
	cout<<cuD.str();
	cout<<matA.dot(matB,true,true).str();

}
void test4(){
	float a[6]={1,2,3,4,5,6};
	float b[12]={1,2,3,4,5,6,7,8,9,10,11,12};
	Matrix matA(3,2,a);
	Matrix matB(4,3,b);	
	CuMatrix cuA(matA,handle);
	CuMatrix cuB(matB,handle);
	CuMatrix cuD=cuA.dot(cuB,true,true);
	cuD.updateMatrixFromGPU();
	cout<<cuD.str();
	cout<<matA.dot(matB,true,true).str();

}

Matrix softmax(Matrix& mat) {
    FeatType* result = new FeatType[mat.getNumElemts()];

    for (unsigned r = 0; r < mat.getRows(); ++r) {
        unsigned length = mat.getCols();
        FeatType* vecSrc = mat.getData() + r * length;
        FeatType* vecDst = result + r * length;

        FeatType denom = 0.0;
        for (unsigned c = 0; c < length; ++c) {
            vecDst[c] = std::exp(vecSrc[c]);
            denom += vecDst[c];
        }

        for (unsigned c = 0; c < length; ++c) {
            vecDst[c] /= denom;
        }
    }

    return Matrix(mat.getRows(), mat.getCols(), result);
}
void test_softmax(){
	float alpha=1.0,beta=0.0;
	float b[ROW*COL];
	Matrix matB(ROW,COL,b);
	for (int i=0;i<ROW*COL;++i){
		matB.getData()[i]=i/ROW*COL;
	}
	CuMatrix cuB(matB,handle);
	
	clock_t t=clock();
	CuMatrix cuC=cu.softmaxRows(cuB);
	t = clock() - t;
  	printf ("GPU %f\n",((float)t)/CLOCKS_PER_SEC);

  	t=clock();
	Matrix C=softmax(matB);
	t = clock() - t;

	printf ("CPU %f\n",((float)t)/CLOCKS_PER_SEC);
	// cout<<C.str();
	cuC.updateMatrixFromGPU();
	// cout<<cuC.str();

	printf("CHECKING\n");
	for (int i=0;i<ROW*COL;++i){
		if(C.getData()[i]-cuC.getData()[i]>0.01){
			printf("ERROR\n");
			return;
		}
	}
}

void test_tanh(){
	float alpha=1.0,beta=0.0;
	float a[ROW*COL];
	float b[ROW*COL];

	Matrix matA(ROW,COL,a);
	Matrix matB(ROW,COL,b);
	for (int i=0;i<ROW*COL;++i){
		matA.getData()[i]=i/ROW*COL;
		matB.getData()[i]=i/ROW*COL;
	}
	CuMatrix cuB(matB,handle);
	
	clock_t t=clock();
		for (unsigned i = 0; i < matA.getNumElemts(); ++i)
        	matA.getData()[i] = std::tanh(matA.getData()[i]);
	t = clock() - t;
  	printf ("CPU %f\n",((float)t)/CLOCKS_PER_SEC);

  	t=clock();
  	cudnnTensorDescriptor_t srcTensorDesc;
	cudnnCreateTensorDescriptor(&srcTensorDesc);
	cudnnSetTensor4dDescriptor(srcTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
            ROW,1,1,COL);

	cudnnActivationDescriptor_t actDesc;
	cudnnCreateActivationDescriptor(&actDesc);
	cudnnSetActivationDescriptor(actDesc,CUDNN_ACTIVATION_TANH,CUDNN_NOT_PROPAGATE_NAN,1.0);
	cudnnActivationForward(cu.cudnnHandle,actDesc,
		&alpha,srcTensorDesc,cuB.devPtr,&beta,srcTensorDesc,cuB.devPtr);
	t = clock() - t;

	printf ("CUDNN %f\n",((float)t)/CLOCKS_PER_SEC);
	cuB.updateMatrixFromGPU();

	printf("CHECKING\n");
	for (int i=0;i<ROW*COL;++i){
		if(matA.getData()[i]-cuB.getData()[i]>0.01){
			printf("ERROR\n");
			return;
		}
	}
}

static Matrix
activateDerivate(Matrix& mat) {
    FeatType *res = new FeatType[mat.getNumElemts()];
    FeatType *zData = mat.getData();

    for (unsigned i = 0; i < mat.getNumElemts(); ++i)
        res[i] = 1 - std::pow(std::tanh(zData[i]), 2);

    return Matrix(mat.getRows(), mat.getCols(), res);
}

// void test_tanh_derivative(){
// 	float alpha=1.0,beta=0.0;
// 	float x[ROW*COL];
// 	float y[ROW*COL];
// 	float dy[ROW*COL];
// 	float dx[ROW*COL];
	

// 	Matrix matX(ROW,COL,x);
// 	Matrix matY(ROW,COL,y);
// 	Matrix matDX(ROW,COL,dx);
// 	Matrix matDY(ROW,COL,dy);

	
// 	for (int i=0;i<ROW*COL;++i){
// 		x[i]=i;
// 		dx[i]=1.;
// 		dy[i]=1.;
// 		y[i] = std::tanh(x[i]);
// 	}
        

// 	clock_t t=clock();
// 	Matrix matC=activateDerivate(matY);
// 	t = clock() - t;
//   	printf ("CPU %f\n",((float)t)/CLOCKS_PER_SEC);

//   	CuMatrix cuX(matX,handle);
//   	CuMatrix cuY(matY,handle);
//   	CuMatrix cuDX(matDX,handle);
//   	CuMatrix cuDY(matDY,handle);

//   	t=clock();
//   	cudnnTensorDescriptor_t srcTensorDesc;
// 	cudnnCreateTensorDescriptor(&srcTensorDesc);
// 	cudnnSetTensor4dDescriptor(srcTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
//             ROW,1,1,COL);

// 	cudnnTensorDescriptor_t dyTensorDesc;
// 	cudnnCreateTensorDescriptor(&srcTensorDesc);
// 	cudnnSetTensor4dDescriptor(srcTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
//             COL,1,1,ROW);

// 	cudnnActivationDescriptor_t actDesc;
// 	cudnnCreateActivationDescriptor(&actDesc);
// 	cudnnSetActivationDescriptor(actDesc,CUDNN_ACTIVATION_TANH,CUDNN_NOT_PROPAGATE_NAN,1.0);
// 	cudnnActivationBackward(cu.cudnnHandle,actDesc,
// 		&alpha,
// 		srcTensorDesc,cuY.devPtr,
// 		dyTensorDesc,cuDY.devPtr,
// 		srcTensorDesc,cuDX.devPtr,
// 		&beta,
// 		srcTensorDesc,cuX.devPtr);
// 	t = clock() - t;

// 	printf ("CUDNN %f\n",((float)t)/CLOCKS_PER_SEC);
// 	cuDY.updateMatrixFromGPU();

// 	printf("CHECKING\n");
// 	for (int i=0;i<ROW*COL;++i){
// 		if(cuDY.getData()[i]-matC.getData()[i]>0.01){
// 			printf("ERROR\n");
// 			return;
// 		}
// 	}
// }


int main()
{	
	handle=cu.handle;
	
	return 0;
}


//********Not that important now since our backward prop 
// computation combines the cross entropy and softmax together
// void test_softmax_back(){
// 	float a[ROW*COL];
// 	float b[ROW*COL];

// 	Matrix matA(ROW,COL,b);
// 	for (int i=0;i<ROW*COL;++i){
// 		matA.getData()[i]=i/ROW*COL;
// 	}
// 	CuMatrix cuA(matA,handle);

// 	Matrix matB(ROW,COL,b);
// 	for (int i=0;i<ROW*COL;++i){
// 		matB.getData()[i]=(ROW*COL-i)/ROW*COL;
// 	}
// 	CuMatrix cuB(matB,handle);

// 	clock_t t=clock();
// 	CuMatrix cuC=cu.hadamardSub(cuA,cuB);
// 	t = clock() - t;
//   	printf ("GPU %f\n",((float)t)/CLOCKS_PER_SEC);
//   	cuC.updateMatrixFromGPU();

//   	t=clock();
//   	CuMatrix res(Matrix(ROW,COL,(FeatType *)NULL),handle);
//     cudnnTensorDescriptor_t srcTensorDesc, sftTensorDesc;
//     cudnnCreateTensorDescriptor(&srcTensorDesc);
//     cudnnCreateTensorDescriptor(&sftTensorDesc);
//     cudnnSetTensor4dDescriptor(srcTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
//             ROW,1,1,COL);
//     cudnnSetTensor4dDescriptor(sftTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
//             ROW,1,1,COL);
// 	cudnnSoftmaxBackward(cu.cudnnHandle,CUDNN_SOFTMAX_ACCURATE,CUDNN_SOFTMAX_MODE_INSTANCE,
// 			&alpha,srcTensorDesc, cuA.devPtr, 
// 			srcTensorDesc, cuB.devPtr, 
//             &beta, srcTensorDesc, res.devPtr);

// 	t = clock() - t;
// 	printf ("CUDNN %f\n",((float)t)/CLOCKS_PER_SEC);
// 	// cout<<C.str();
// 	res.updateMatrixFromGPU();
// 	// cout<<cuC.str();

// 	printf("CHECKING\n");
// 	for (int i=0;i<ROW*COL;++i){
// 		if(cuC.getData()[i]-res.getData()[i]>0.001){
// 			printf("ERROR %d\n",i);
// 			return;
// 		}
// 	}
// }
