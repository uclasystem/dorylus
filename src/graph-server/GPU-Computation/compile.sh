#!/bin/bash
if [[ $LD_LIBRARY_PATH == '' ]]; then
	echo "LD_LIBRARY_PATH was not set yet.\n Make Sure 'source' is used.\n "
	curr=`pwd`
	export LD_LIBRARY_PATH=`pwd`
	cd $curr/../../common
	export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:`pwd`
	cd $curr
fi
a=`pwd`
cd ../../common
nvcc -std=c++11 -shared --compiler-options "-fPIC"  matrix.cpp  -I. -o libmatrix.so -lcblas -lopenblas
cd $a
nvcc -shared --compiler-options "-fPIC -std=c++11" cu_matrix.cu -L../../common -lmatrix -o libcumatrix.so -lcublas -lcudnn -lcblas  -lopenblas 
nvcc -shared --compiler-options "-fPIC -std=c++11" comp_unit.cu -L../../common -lmatrix -o libcu.so -lcublas -lcudnn -lcblas  -lopenblas
nvcc test.cpp -L. -L../../common  -lcu -lcumatrix -lmatrix -lcblas -lcublas -lopenblas -lcudnn -lcusparse

