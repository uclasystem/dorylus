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

#include "resource_comm.hpp"
#include "../GPU-Computation/comp_server.cuh"
#include "../utils/utils.hpp"
#include "../../common/matrix.hpp"
#include "../../common/utils.hpp"


/**
 *
 * Communicate with local GPU process using IPC 
 * 
 */
class GPUComm : public ResourceComm{

public:

    GPUComm(unsigned nodeId_, unsigned numNodes_, unsigned dataserverPort_,const std::string& wServersFile,unsigned wPort_);

    // For forward-prop.
    void newContextForward(FeatType *dataBuf, FeatType *zData_, FeatType *actData_,
                              unsigned numLocalVertices_, unsigned numFeats, unsigned numFeatsNext_, bool eval_);
    void requestForward(unsigned layer);
    void waitLambdaForward(){};
    void invokeLambdaForward(unsigned layer, unsigned lambdaId){};

    // For backward-prop.
    void newContextBackward(FeatType **zBufs, FeatType **actBufs, FeatType *targetBuf,
                            unsigned numLocalVertices, std::vector<unsigned> layerConfig);
    void requestBackward(unsigned numLayers);
    void sendBackpropChunks();

    //for validation
    void setTrainValidationSplit(float trainPortion, unsigned numLocalVertices);
    //cannot be called if newContextBackward is never called due to the assignment of targetmatrix
    void sendTargetMatrix();

    // Send a message to the coordination server to shutdown.
    void sendShutdownMessage();

    template <class T>
    T requestFourBytes();
    void sendFourBytes(char* data);
    void sendMatrix(Matrix& m);
    void sendResultPtr(FeatType* ptr);
    friend class ComputingServer;

private:
    unsigned nodeId;
    unsigned numNodes;
    unsigned numLocalVertices;

    std::string wServersFile;

    std::thread comp_server_thread;

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

    bool eval;
    float split;

    //backward 
    std::vector<Matrix> zMatrices;
    std::vector<Matrix> actMatrices;
    Matrix targetMatrix;

    zmq::message_t confirm;

};




#endif // GPU_COMM_HPP