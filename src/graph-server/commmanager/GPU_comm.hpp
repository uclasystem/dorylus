#ifndef __GPU_COMM_HPP__
#define __GPU_COMM_HPP__


#include "../GPU-Computation/comp_server.cuh"

#include <chrono>
#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <zmq.hpp>
#include "resource_comm.hpp"
#include "../utils/utils.hpp"
#include "../../common/matrix.hpp"
#include "../../common/utils.hpp"

class ComputingServer;

/**
 *
 * Communicate with local GPU process using IPC
 *
 */
class GPUComm : public ResourceComm{

public:

    GPUComm(unsigned nodeId_, unsigned numNodes_, unsigned dataserverPort_,const std::string& wServersFile,unsigned wPort_,unsigned totalLayers_);

    // For forward-prop.
    void newContextForward(unsigned layer, FeatType *dataBuf, FeatType *zData_, FeatType *actData_,
                              unsigned numLocalVertices_, unsigned numFeats, unsigned numFeatsNext_);
    void requestForward(unsigned layer, bool lastLayer);
    void waitLambdaForward(unsigned layer, bool lastLayer) {};
    void invokeLambdaForward(unsigned layer, unsigned lambdaId, bool lastLayer) {};


    // For backward-prop.

    void newContextBackward(unsigned layer, FeatType *oldGradBuf, FeatType *newGradBuf, std::vector<Matrix> *savedTensors, FeatType *targetBuf,
                            unsigned numLocalVertices, unsigned inFeatDim, unsigned outFeatDim, unsigned targetDim);
    void requestBackward(unsigned layer, bool lastLayer);
    void invokeLambdaBackward(unsigned layer, unsigned lambdaId, bool lastLayer) {};
    void waitLambdaBackward(unsigned layer, bool lastLayer) {}

    //for validation
    void setTrainValidationSplit(float trainPortion, unsigned numLocalVertices);
    //cannot be called if newContextBackward is never called due to the assignment of targetmatrix
    void sendTargetMatrix();

    // Send a message to the coordination server to shutdown.
    void sendShutdownMessage();

    friend class ComputingServer;

private:
    unsigned totalLayers;
    unsigned nodeId;
    unsigned numNodes;
    unsigned numLocalVertices;

    unsigned currLayer;

    std::string wServersFile;

    //ntw related objs
    zmq::context_t ctx;
    zmq::socket_t dataSocket;
    unsigned dPort;
    unsigned wPort;

    //forward
    //data related objs
    Matrix actMatrix;   // Current layer's feats.
    FeatType *zData;    // Places to store the results from lambda.
    FeatType *actData;
    unsigned numFeatsNext;

    float split;

    //backward
    Matrix oldGradMatrix;
    Matrix newGradMatrix;
    Matrix targetMatrix;
    std::vector<Matrix> *savedTensors;

    ComputingServer* comp_server;

};


#endif // GPU_COMM_HPP
