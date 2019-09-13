#ifndef __GPU_COMM_HPP__
#define __GPU_COMM_HPP__

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <zmq.hpp>

#include "../utils/utils.hpp"
#include "../../common/matrix.hpp"
#include "../../common/utils.hpp"


/**
 *
 * Communicate with local GPU process using IPC
 * 
 */
class GPUComm {

public:

    GPUComm(unsigned nodeId_,unsigned dataserverPort_):
        nodeId(nodeId_),
        dPort(dataserverPort_),
        dataSocket(ctx,ZMQ_REQ){

        char ipc_addr[50];
        sprintf(ipc_addr, "ipc:///tmp/GPU_COMM:%u", dPort);
        dataSocket.connect(ipc_addr);

        zmq::message_t confirm(5);
        zmq::message_t init_header(HEADER_SIZE);
        populateHeader((char*)init_header.data(),nodeId);
        dataSocket.send(init_header);
        dataSocket.recv(&confirm);
    }
    // For forward-prop.
    void newContextForward(FeatType *dataBuf, FeatType *zData_, FeatType *actData_,
                              unsigned numLocalVertices, unsigned numFeats, unsigned numFeatsNext_);
    void requestForward(unsigned layer);

    // Send a message to the coordination server to shutdown.
    void sendShutdownMessage();


private:
    //ntw related objs
    zmq::context_t ctx;
    zmq::socket_t dataSocket;
    unsigned dPort;

    //data related objs
    Matrix actMatrix;   // Current layer's feats.
    FeatType *zData;    // Places to store the results from lambda.
    FeatType *actData;
    unsigned numFeatsNext;

    unsigned nodeId;

};




#endif // GPU_COMM_HPP
