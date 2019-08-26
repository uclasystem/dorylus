#include "comp_server/comp_server.hpp"
#include "comp_unit/comp_unit.hpp"
#include <iostream>
#include <cstring>
#include <cstdio>
#include <sstream>
#include <chrono>
#include "../../src/utils/utils.hpp"


int main(int argc, char const *argv[])
{
	// Matrix A=Matrix(3,2,new float[100]{1,2,3,4,5,6});
	// Matrix B=Matrix(2,3,new float[100]{10,100,1000,11,111,1111});
	// Matrix C=Matrix(3,3,new float[]);
	

	// Matrix A=Matrix(10,10,new float[100]);
	// Matrix B=Matrix(10,10,new float[100]);
	// Matrix C=Matrix(10,10,new float[100]);
	// for(int i=0;i<100;++i){
	// 	(A.getData())[i]=-0.9;
	// 	(B.getData())[i]=1;
	// }
	// ComputingUnit cu=ComputingUnit();
	// C=cu.dot(A,B);
	// printf("dot %s\n", C.str().c_str());
	// cu.activate(A);
	// printf("activate %s\n", A.str().c_str());


	ComputingServer cs=ComputingServer(8123,"./wserverip",65432);
	cs.run();
	return 0;
}
