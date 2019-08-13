#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

using namespace std::chrono;

class ComputingUnit
{
public:
	ComputingUnit();
	Matrix compute();
	Matrix dot();
	Matrix activate();
	~ComputingUnit();
};