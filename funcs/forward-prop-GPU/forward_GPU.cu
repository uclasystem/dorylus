// #include "comp_unit/comp_unit.hpp"
#include "comp_server/comp_server.hpp"
#include <iostream>
#include <cstring>
#include <cstdio>
#include "../../src/utils/utils.hpp"
#include <sstream>

int main(int argc, char const *argv[])
{

	// Matrix A=Matrix(3,2,new float[6]{1,2,3,4,5,6});
	// Matrix B=Matrix(2,3,new float[6]{10,100,1000,11,111,1111});
	// Matrix C=Matrix(3,3,new float[9]);
	// ComputingUnit cu=ComputingUnit();
	// C=cu.dot(A,B);
	// cu.activate(A);
	ComputingServer cs=ComputingServer(2000,"0.0.0.0",20001);

	


	return 0;
}