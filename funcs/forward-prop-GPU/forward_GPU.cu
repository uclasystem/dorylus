#include <iostream>
#include <cstring>
#include <cstdio>
#include <sstream>
#include <chrono>
#include "../../src/utils/utils.hpp"
#include "comp_server/comp_server.hpp"
#include <math.h>

#define ERROR 0.01
int compare(float* A, float * B,unsigned size){
    for(int i= 0 ; i < size;++i){
		if(abs(A[i]-B[i])>0.01){
			printf("WRONG RESULT\n");
			exit(1);
		}
				
	}
	return 0;
}

int main(int argc, char *argv[])
{
	const unsigned SIZE=10000;
	const unsigned ROW=100;
	const unsigned COL=100;
	//GEMM 
	ComputingUnit cu;
	float a[]={1,1,1,2,2,2};
	float b[]={10,10,10,10,10,20};
	Matrix A(2,3,a);
	Matrix B(3,2,b);
	CuMatrix C=cu.dot(A,B);
	std::cout<<C.str();


	//tanf
	//*****SPEED********
	float d[10000];
	for(int i=0;i<10000;++i)
		d[i]=i/10;
	Matrix D(ROW,COL,d);
	CuMatrix cuD(D,cu.handle);
    double t1=getTimer();
	cu.activate(cuD);
    printf("tanf gpu*: %lf\n", getTimer()-t1);
	// std::cout<<cuD.str();

	t1=getTimer();
	for(int i= 0 ; i < SIZE;++i){
		d[i]=tanf(d[i]);
	}
    printf("tanf cpu*: %lf\n", getTimer()-t1);
   	compare(cuD.getData(),d,SIZE);

   	//softmax
   	float e[10]={1,2,3,4,5,1,2,3,4,6};
   	Matrix E(2,5,e);
   	cu.softmaxRows(E);



	return 0;
}

// int main(int argc, char *argv[])
// {
// 	assert(argc==4);
// 	unsigned dataPort=atoi(argv[1]);
// 	char *wserverIp=argv[2];
// 	unsigned wserverPort=atoi(argv[3]);
// 	ComputingServer cs=ComputingServer(dataPort,wserverIp,wserverPort);
// 	cs.run();
// 	return 0;
// }
