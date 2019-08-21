#include "GPU_comm.hpp"

int main()
{
	auto gc=GPUComm(1,1234);
	gc.requestForward(1);
	return 0;
}
