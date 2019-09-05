#include <iostream>
#include <cstring>
#include <cstdio>
#include <sstream>
#include <chrono>
#include "../../src/utils/utils.hpp"
#include "comp_server/comp_server.hpp"


int main(int argc, char *argv[])
{
	assert(argc==4);
	unsigned dataPort=atoi(argv[1]);
	char *wserverIp=argv[2];
	unsigned wserverPort=atoi(argv[3]);
	ComputingServer cs=ComputingServer(dataPort,wserverIp,wserverPort);
	cs.run();
	return 0;
}



// #include <math.h>
// #include <cblas.h>
// #define ERROR 0.01
// int compare(float* A, float * B,unsigned size){
//     for(int i= 0 ; i < size;++i){
// 		if(abs(A[i]-B[i])>ERROR){
// 			printf("%f\n", abs(A[i]-B[i]));
		
// 		}
				
// 	}
// 	return 0;
// }

// Matrix
// dotGDwithWTrans(Matrix& matLeft, Matrix& matRight) {
//     unsigned m = matLeft.getRows(), k = matLeft.getCols(), n = matRight.getRows();
//     assert(k == matRight.getCols());

//     FeatType *res = new FeatType[m * n];
//     cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, 1.0,
//                 matLeft.getData(), k, matRight.getData(), k, 0.0, res, n);

//     return Matrix(m, n, res);
// }

// Matrix
// dotActTranswithGD(Matrix& matLeft, Matrix& matRight, float learning_rate) {
//     unsigned m = matLeft.getCols(), k = matLeft.getRows(), n = matRight.getCols();
//     assert(k == matRight.getRows());

//     FeatType *res = new FeatType[m * n];
//     cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, m, n, k, learning_rate,
//                 matLeft.getData(), m, matRight.getData(), n, 0.0, res, n);

//     return Matrix(m, n, res);
// }

// Matrix
// activateDerivate(Matrix& mat) {
//     FeatType *res = new FeatType[mat.getNumElemts()];
//     FeatType *zData = mat.getData();

//     for (unsigned i = 0; i < mat.getNumElemts(); ++i)
//         res[i] = 1 - std::pow(std::tanh(zData[i]), 2);

//     return Matrix(mat.getRows(), mat.getCols(), res);
// }

// Matrix
// softmaxRows(Matrix& mat) {
//     FeatType *res = new FeatType[mat.getNumElemts()];

//     for (unsigned i = 0; i < mat.getRows(); ++i) {
//         unsigned length = mat.getCols();
//         FeatType *vecSrc = mat.getData() + i * length;
//         FeatType *vecDst = res + i * length;

//         FeatType denom = 0.;
//         for (unsigned j = 0; j < length; ++j) {
//             vecDst[j] = std::exp(vecSrc[j]);
//             denom += vecDst[j];
//         }
//         for (unsigned j = 0; j < length; ++j)
//             vecDst[j] /= denom;
//     }

//     return Matrix(mat.getRows(), mat.getCols(), res);
// }

// Matrix
// hadamardSub(Matrix& matLeft, Matrix& matRight) {
//     assert(matLeft.getRows() == matRight.getRows());
//     assert(matLeft.getCols() == matRight.getCols());
    
//     FeatType *res = new FeatType[matLeft.getNumElemts()];
//     FeatType *leftData = matLeft.getData(), *rightData = matRight.getData();

//     for (unsigned i = 0; i < matLeft.getNumElemts(); ++i)
//         res[i] = leftData[i] - rightData[i];

//     return Matrix(matLeft.getRows(), matRight.getCols(), res);
// }

// Matrix
// hadamardMul(Matrix& matLeft, Matrix& matRight) {
//     assert(matLeft.getRows() == matRight.getRows());
//     assert(matLeft.getCols() == matRight.getCols());
    
//     FeatType *res = new FeatType[matLeft.getNumElemts()];
//     FeatType *leftData = matLeft.getData(), *rightData = matRight.getData();

//     for (unsigned i = 0; i < matLeft.getNumElemts(); ++i)
//         res[i] = leftData[i] * rightData[i];

//     return Matrix(matLeft.getRows(), matRight.getCols(), res);
// }

// int main(int argc, char *argv[])
// {
// 	const unsigned SIZE=1000000;
// 	const unsigned ROW=1000;
// 	const unsigned COL=1000;
	
// 	ComputingUnit cu;
// 	double t1;
	
	// printf("GEMM\n");
	// float a[]={1,1,1,2,2,2};
	// float b[]={10,10,10,10,10,10,10,10,10,10,10,10};
	// Matrix A(2,3,a);
	// Matrix B(3,4,b);
	// CuMatrix C=cu.dot(A,B);
	// std::cout<<C.str();


	// //tanf
	// //*****SPEED********
	// float d[SIZE];
	// for(int i=0;i<SIZE;++i)
	// 	d[i]=i/10;
	// Matrix D(ROW,COL,d);
	// CuMatrix cuD(D,cu.handle);
 //    t1=getTimer();
	// cu.activate(cuD);
 //    printf("tanf gpu*: %lf\n", getTimer()-t1);
	// // std::cout<<cuD.str();

	// t1=getTimer();
	// for(int i= 0 ; i < SIZE;++i){
	// 	d[i]=tanf(d[i]);
	// }
 //    printf("tanf cpu*: %lf\n", getTimer()-t1);
 //   	compare(cuD.getData(),d,SIZE);

 //   	//softmax
 //   	printf("softmax:  \n");
 //   	float e[SIZE];
 //   	for(int i=0;i<SIZE;++i)
	// 	e[i]=i/10;
 //   	Matrix E(ROW,COL,e);
 //   	CuMatrix cuE(E,cu.handle);
 //   	t1=getTimer();
 //   	CuMatrix  cuF=cu.softmaxRows(cuE);
 //   	printf("softmax gpu*: %lf\n", getTimer()-t1);
 //   	cuF.updateMatrixFromGPU();
 //   	t1=getTimer();
 //   	Matrix F=softmaxRows(E);
 //   	printf("softmax cpu*: %lf\n", getTimer()-t1);
 //   	compare(cuF.getData(),F.getData(),F.getNumElemts());


 //   	printf("hadamardSub:  \n");
 //   	float j[SIZE];
	// float g[SIZE];
	// for(unsigned i=0;i<SIZE;++i){
	// 	g[i]=sin(i);
	// 	j[i]=cos(i);
	// }
	// Matrix J(ROW,COL,j);
	// Matrix G(ROW,COL,g);
	// CuMatrix cuJ(J,cu.handle);
	// CuMatrix cuG(G,cu.handle);
 //   	t1=getTimer();
 //   	CuMatrix  cuH=cu.hadamardSub(cuJ,cuG);
 //   	printf("hadamardSub gpu*: %lf\n", getTimer()-t1);
 //   	t1=getTimer();
 //   	Matrix H=hadamardSub(J,G);
 //   	printf("hadamardSub cpu*: %lf\n", getTimer()-t1);
 //   	cuH.updateMatrixFromGPU();
 //   	compare(cuH.getData(),H.getData(),SIZE);
	
 //  	printf("hadamardMul:  \n");
 //   	float j[SIZE];
	// float g[SIZE];
	// for(unsigned i=0;i<SIZE;++i){
	// 	g[i]=sin(i);
	// 	j[i]=cos(i);
	// }
	// Matrix J(ROW,COL,j);
	// Matrix G(ROW,COL,g);
	// CuMatrix cuJ(J,cu.handle);
	// CuMatrix cuG(G,cu.handle);
 //   	t1=getTimer();
 //   	CuMatrix  cuH=cu.hadamardMul(cuJ,cuG);
 //   	printf("hadamardMul gpu*: %lf\n", getTimer()-t1);
 //   	t1=getTimer();
 //   	Matrix H=hadamardMul(J,G);
 //   	printf("hadamardMul cpu*: %lf\n", getTimer()-t1);
 //   	cuH.updateMatrixFromGPU();
 //   	compare(cuH.getData(),H.getData(),SIZE);


	// printf("activateDerivate:  \n");
 //   	float j[SIZE];
	// for(unsigned i=0;i<SIZE;++i)
	// 	j[i]=sin(i);
	// Matrix J(ROW,COL,j);
	// CuMatrix cuJ(J,cu.handle);
 //   	t1=getTimer();
 //   	CuMatrix  cuH=cu.activateDerivate(cuJ);
 //   	printf("activateDerivate gpu*: %lf\n", getTimer()-t1);
 //   	t1=getTimer();
 //   	Matrix H=activateDerivate(J);
 //   	printf("activateDerivate cpu*: %lf\n", getTimer()-t1);
 //   	cuH.updateMatrixFromGPU();
 //   	compare(cuH.getData(),H.getData(),SIZE);

	// printf("dotActTranswithGD:  \n");
	// printf("Speed\n");
 //   	float j[SIZE];
	// float g[SIZE];
	// for(unsigned i=0;i<SIZE;++i){
	// 	g[i]=sin(i);
	// 	j[i]=cos(i);
	// }
	// Matrix J(ROW,COL,j);
	// Matrix G(ROW,COL,g);
	// CuMatrix cuJ(J,cu.handle);
	// CuMatrix cuG(G,cu.handle);
 //   	t1=getTimer();
 //   	CuMatrix  cuH=cu.dotActTranswithGD(cuJ,cuG,0.01);
 //   	printf("dotActTranswithGD gpu*: %lf\n", getTimer()-t1);
 //   	t1=getTimer();
 //   	Matrix H=dotActTranswithGD(J,G,0.01);
 //   	printf("dotActTranswithGD cpu*: %lf\n", getTimer()-t1);
 //   	cuH.updateMatrixFromGPU();
 //   	compare(cuH.getData(),H.getData(),SIZE);

 //   	printf("Acc\n");
 //   	float j[6];
	// float g[12];
	// for(unsigned i=0;i<6;++i)
	// 	j[i]=i+1;
	// for(unsigned i=0;i<12;++i)
	// 	g[i]=i/2;
	// Matrix J(3,2,j);
	// Matrix G(3,4,g);
	// CuMatrix cuJ(J,cu.handle);
	// CuMatrix cuG(G,cu.handle);
 //   	t1=getTimer();
 //   	CuMatrix  cuH=cu.dotActTranswithGD(cuJ,cuG,0.01);
 //   	printf("dotActTranswithGD gpu*: %lf\n", getTimer()-t1);
 //   	t1=getTimer();
 //   	Matrix H=dotActTranswithGD(J,G,0.01);
 //   	printf("dotActTranswithGD cpu*: %lf\n", getTimer()-t1);
 //   	cuH.updateMatrixFromGPU();
 //   	compare(cuH.getData(),H.getData(),8);
 //  	printf("CPU %s\n", H.str().c_str());
 //   	printf("GPU %s\n", cuH.str().c_str());


	// printf("dotGDwithWTrans:  \n");
 //   	float j[6];
	// float g[12];
	// for(unsigned i=0;i<6;++i)
	// 	j[i]=sin(i);
	// for(unsigned i=0;i<12;++i)
	// 	g[i]=cos(i);

	// Matrix J(2,3,j);
	// Matrix G(4,3,g);
	// CuMatrix cuJ(J,cu.handle);
	// CuMatrix cuG(G,cu.handle);
 //   	t1=getTimer();
 //   	CuMatrix  cuH=cu.dotGDwithWTrans(cuJ,cuG);
 //   	printf("dotGDwithWTrans gpu*: %lf\n", getTimer()-t1);
 //   	t1=getTimer();
 //   	Matrix H=dotGDwithWTrans(J,G);
 //   	printf("dotGDwithWTrans cpu*: %lf\n", getTimer()-t1);
 //   	cuH.updateMatrixFromGPU();
 //   	compare(cuH.getData(),H.getData(),2*4);
 //   	printf("CPU %s\n", H.str().c_str());
 //   	printf("GPU %s\n", cuH.str().c_str());


	// printf("Transpose\n");
	// float x[8];
	// for (int i = 0; i < 8; ++i)
	// 	x[i]=(float)i;
	// Matrix m(2,4,x);
	// std::cout<<m.str();
	// CuMatrix cuM(m,cu.handle);

	// CuMatrix MT=cuM.transpose();
	// MT.updateMatrixFromGPU();
	// std::cout<<MT.str();


// 	return 0;
// }


