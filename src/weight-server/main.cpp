#include <cassert>
#include <iostream>
#include <string>
#include <mutex>
#include "weightserver.hpp"

/** Main entrance: Starts a weightserver instance and run. */
int
main(int argc, char *argv[]) {
// TODO: May need to start using an arg parser like boost.
    assert(argc == 7);
    std::string weightServersFile = argv[1];
    std::string myPrIpFile = argv[2];
    unsigned serverPort = std::atoi(argv[3]);
    unsigned listenerPort = std::atoi(argv[4]);
    std::string configFileName = argv[5];
    // Set output file location. Still needs to append nodeId.
    std::string tmpFileName = std::string(argv[6]) + "/output_";
    bool sync = (bool)(std::atoi(argv[7]));

    WeightServer ws(weightServersFile, myPrIpFile, listenerPort, configFileName, serverPort, tmpFileName, sync);

    // Run in a detached thread because so that we can wait
    // on a condition variable.
    ws.run();

    // Wait for one of the threads to mark the finished bool true
    // then end the main thread.
    std::unique_lock<std::mutex> lk(ws.termMtx);
    ws.termCV.wait(lk, [&] { return ws.term; });
    std::cerr << "We are terminating the weight server" << std::endl;

    return 0;
}
