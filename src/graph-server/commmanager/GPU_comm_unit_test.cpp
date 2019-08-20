#include "GPU_comm.hpp"

int main()
{
	auto gc=GPUComm(1,1234,2234);
	gc.requestForward(2);
	return 0;
}
