#!/bin/bash
a=`pwd`
cd ../../common
nvcc -std=c++11 -shared --compiler-options "-fPIC"  matrix.cpp  -I. -o libmatrix.so -lcblas -lopenblas
cd $a
nvcc -shared --compiler-options "-fPIC -std=c++11" cu_matrix.cu -L../../common -lmatrix -o libcumatrix.so -lcublas -lcudnn -lcblas  -lopenblas 
nvcc -shared --compiler-options "-fPIC -std=c++11" comp_unit.cu -L../../common -lmatrix -o libcu.so -lcublas -lcudnn -lcblas  -lopenblas
nvcc test.cpp -L. -L../../common  -lcu -lcumatrix -lmatrix -lcblas -lcublas -lopenblas -lcudnn
