#include <cassert>
#include <iostream>
#include <string>
#include <mutex>
#include "weightserver.hpp"

/** Main entrance: Starts a weightserver instance and run. */
int
main(int argc, char *argv[]) {
// TODO: May need to start using an arg parser like boost.
    assert(argc >= 13);
    std::string wserverFile = argv[1];
    std::string myPrIpFile = argv[2];
    std::string gserverFile = argv[3];
    unsigned serverPort = std::atoi(argv[4]);
    unsigned listenerPort = std::atoi(argv[5]);
    unsigned gport = std::atoi(argv[6]);
    std::string configFile = argv[7];
    // Set output file location. Still needs to append nodeId.
    std::string tmpFile = std::string(argv[8]) + "/output_";
    bool sync = (bool)(std::atoi(argv[9]));
    float targetAcc = std::atof(argv[10]);
    bool block = (bool)(std::atoi(argv[11])); // for CPU/GPU
    float learning_rate = std::atof(argv[12]);

    WeightServer ws(wserverFile, myPrIpFile, gserverFile,
                    listenerPort, serverPort, gport,
                    configFile, tmpFile,
                    sync, targetAcc, block,
                    learning_rate);

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
