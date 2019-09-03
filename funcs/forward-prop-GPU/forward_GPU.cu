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
