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
#include "../../utils/utils.hpp"



/**
 *
 * Communicate with local GPU process using TCP
 * 
 */
class GPUComm {

public:

    GPUComm(unsigned nodeId_,unsigned dataserverPort_):
        nodeId(nodeId_),
        dataserverPort(dataserverPort_),
        dataSocket(ctx,ZMQ_REQ){

        char dataIPCAddress[50];
        sprintf(dataIPCAddress, "ipc:///tmp/feeds/%u", dataserverPort);
        int ret = dataSocket.connect(dataIPCAddress);
        if (ret != 0 ){ printLog(nodeId,"IPC connect to %s failed",dataIPCAddress);}

    }
    // For forward-prop.
    void requestForward(unsigned layer);

    // Send a message to the coordination server to shutdown.
    void sendShutdownMessage();


private:
    //ntw related objs
    zmq::context_t ctx;
    zmq::socket_t dataSocket;
    unsigned dataserverPort;

    //data related objs
    FeatType *zData;
    FeatType *actData;
    Matrix *actMatrix;
    unsigned numFeatsNext;


    unsigned nodeId;

};

void GPUComm::requestForward(unsigned layer){

}

void GPUComm::sendShutdownMessage(){


}



#endif // GPU_COMM_HPP
