#ifndef _COORD_SERVER_HPP__
#define _COORD_SERVER_HPP__


#include <chrono>
#include <iostream>
#include <fstream>
#include <mutex>
#include <memory>
#include <sstream>
#include <thread>
#include <vector>
#include <sys/stat.h>
#include <unistd.h>
#include <zmq.hpp>
#include <boost/algorithm/string/trim.hpp>

#include "../common/utils.hpp"


/**
 *
 * Class of the coordserver. Coordination server keeps listening on the dataserver's request of issuing lambda threads.
 * 
 */
class CoordServer {

public:

    CoordServer(char *coordserverPort_, char *weightserverFile_, char *weightserverPort_, char *dataserverPort_);

    // Runs the coordserver, keeps listening on dataserver's requests for lambda threads invocation.
    void run();

private:

    void loadWeightServers(std::vector<char *>& addresses, const std::string& wServersFile);
    void sendInfoMessage(zmq::socket_t& weightsocket, unsigned numLambdas);
    void sendShutdownMessage(zmq::socket_t& weightsocket);

    char *coordserverPort;
    char *weightserverFile;
    char *weightserverPort;
    char *dataserverPort;
    std::vector<char*> weightserverAddrs;

    zmq::context_t ctx;
};


#endif
