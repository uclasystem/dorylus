#include "cu_matrix.cuh"
#include "comp_unit.cuh"
#include <iostream>

using namespace std;
cublasHandle_t handle;
ComputingUnit cu;
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
void test5(){
	float b[12]={1,2,3,4,5,6,7,8,9,10,11,12};
	
	Matrix matB(4,3,b);	
	CuMatrix cuB=cu.wrapMatrix(matB);
	CuMatrix cuC=cu.softmaxRows(cuB);
	Matrix C=softmax(matB);
	cout<<C.str();
	cuC.updateMatrixFromGPU();
	cout<<cuC.str();
	// cout<<matA.dot(matB,true,true).str();

}
int main()
{	
	handle=cu.handle;
	test5();
	

	return 0;
}
