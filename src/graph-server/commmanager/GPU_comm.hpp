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


void doNotFreeBuffer(void *data, void *hint){
    printf("Buffer is not freed :)\n");
}
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
        // dataSocket.connect("tcp://0.0.0.0:1234");
    }
    // For forward-prop.
    void requestForward(unsigned layer);

    // Send a message to the coordination server to shutdown.
    void sendShutdownMessage();


private:
    //ntw related objs
    zmq::context_t ctx;
    zmq::socket_t dataSocket;
    unsigned dPort;

    //data related objs
    FeatType *zData;
    FeatType *actData;
    Matrix *actMatrix;
    unsigned numFeatsNext;


    unsigned nodeId;

};

void GPUComm::requestForward(unsigned layer){
    try {
        std::string weightIp("0.0.33.3");
        zmq::message_t confirm(5);

        unsigned ROWS=10;
        unsigned COLS=10;

        zmq::message_t header(HEADER_SIZE);
        populateHeader((char *) header.data(), OP::REQ_FORWARD, layer,ROWS,COLS);
        dataSocket.send(header);
        dataSocket.recv(&confirm);

        // Check to make sure that the bounds of this partition do not exceed the bounds of the data array.
        // If they do, set partition end to the end of the array.
        // unsigned partRows = std::ceil((float) actMatrix.getRows() / (float) numLambdasForward);
        // unsigned thisPartRows = partRows;
        // if ((partId * partRows + partRows) > actMatrix.getRows())
        //     thisPartRows = partRows - (partId * partRows + partRows) + actMatrix.getRows();
        // unsigned bufSize = thisPartRows * actMatrix.getCols() * sizeof(FeatType);
        // FeatType *partitionStart = actMatrix.getData() + (partId * partRows * actMatrix.getCols());

        unsigned bufSize = ROWS * COLS * sizeof(FeatType);
        FeatType *partitionStart = new FeatType[bufSize];
        for (int i=0;i<bufSize;++i)
            partitionStart[i]=i;
        printf("Buffer size %u\n", bufSize);
        zmq::message_t partitionData(partitionStart, bufSize, doNotFreeBuffer, NULL);
        dataSocket.send(partitionData);
        //block until computation finish
        dataSocket.recv(&confirm);



    }
    catch(std::exception& ex){
        std::cerr << "[ERROR] " << ex.what() << std::endl;
    }
}


void GPUComm::sendShutdownMessage(){


}



#endif // GPU_COMM_HPP
