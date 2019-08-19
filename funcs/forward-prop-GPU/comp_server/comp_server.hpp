#ifndef __COMP_SERVER_HPP__
#define __COMP_SERVER_HPP__

#include <zmq.hpp>
#include <chrono>
#include <iostream>
#include <fstream>
// #include "../comp_unit/comp_unit.hpp"
#include <string>
#include <cstring>
#include <sstream>
#include "../../../src/utils/utils.hpp"

class ComputingServer {
public:
    ComputingServer(unsigned dPort_,std::string weightServerIp_, unsigned wPort_):
        dPort(dPort_),
        weightServerIp(weightServerIp_),
        wPort(wPort_),
        dataSocket(ctx, ZMQ_REP),
        weightSocket(ctx, ZMQ_REQ)
    {

    }

    // Keep listening to computing requests
    void run();

    // Sending and Requesting functions
    Matrix requestMatrix(zmq::socket_t& socket, int32_t id);
    void sendMatrices(Matrix& zResult, Matrix& actResult, zmq::socket_t& socket, int32_t id);


private:
    //ntw related objs
    zmq::context_t ctx;
    zmq::socket_t dataSocket;
    zmq::socket_t weightSocket;
    
    std::string weightServerIp;
    unsigned dPort;
    unsigned wPort;
    
    // ComputingUnit cu;
};

#endif 
