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

	ComputingServer cs=ComputingServer(8123,"./wserverip",65432);
	cs.run();
	return 0;
}
