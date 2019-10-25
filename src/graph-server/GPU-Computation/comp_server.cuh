#ifndef __COMP_SERVER_HPP__
#define __COMP_SERVER_HPP__

#include <zmq.hpp>
#include <chrono>
#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <sstream>
#include <boost/algorithm/string/trim.hpp>
#include <vector>
#include <condition_variable>
#include "comp_unit.cuh"
#include "../utils/utils.hpp"
#include "../commmanager/GPU_comm.hpp"

class GPUComm;

const float LEARNING_RATE=0.01;
unsigned nodeId;
/** Struct for wrapping over the returned matrices. */
typedef struct {
    std::vector<Matrix> zMatrices;          // Layer 1 -> last.
    std::vector<Matrix> actMatrices;        // Layer 0 -> last.
    Matrix targetMatrix;
} GraphData;



class MessageService{
public:
    MessageService(){};
    MessageService(zmq::context_t& dctx,unsigned dPort_,unsigned wPort_);

    //weight server related
    void setUpWeightSocket(char* addr);
    Matrix requestWeightsMatrix(unsigned layer, OP op=OP::PULL_FORWARD);
    void sendInfoMessage(unsigned numLambdas);
    std::vector<Matrix> requestWeightsMatrices(unsigned numLayers);
    void sendWeightUpdate(Matrix& matrix,unsigned layer);
    void terminateWeightServers(std::vector<char*>& weightServerAddrs);
    void sendShutdownMessage(zmq::socket_t& weightsocket);

    //data server related
    template <class T>
    T requestFourBytes();
    Matrix requestMatrix();
    FeatType* requestResultPtr();
    void sendFourBytes(char* data);
    GraphData requestForwardMatrices(unsigned numLayers);

private:
    zmq::context_t wctx;
    zmq::socket_t* dataSocket;
    zmq::socket_t* weightSocket;
    zmq::message_t confirm;
    
    unsigned wPort;
    bool wsocktReady;
};

class ComputingServer {
public:

    ComputingServer(GPUComm* gpu_comm);

    // Keep listening to computing requests
    void run();

    //For forward
    void processForward();

    //For validation
    void evaluateModel(Matrix& activations);

    //For backward
    void processBackward();
    void gradLayer(unsigned layer);
    void gradLoss(unsigned layer);

private:

    std::vector<char*> weightServerAddrs;
    MessageService msgService;
    ComputingUnit cu;
};


#endif 
