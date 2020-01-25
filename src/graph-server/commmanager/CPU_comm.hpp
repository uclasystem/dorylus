#ifndef __CPU_COMM_HPP__
#define __CPU_COMM_HPP__


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
#include <thread>
#include <boost/algorithm/string/trim.hpp>
#include <fstream>

class MessageServiceCPU {
  public:
    MessageServiceCPU() {};
    MessageServiceCPU(unsigned wPort_, unsigned nodeId_);

    //weight server related
    void setUpWeightSocket(char *addr);
    void prefetchWeightsMatrix(unsigned totalLayers);
    void sendInfoMessage(unsigned numLambdas);

    void sendWeightUpdate(Matrix &matrix, unsigned layer);
    void terminateWeightServers(std::vector<char *> &weightServerAddrs);
    void sendShutdownMessage(zmq::socket_t &weightsocket);

    Matrix getWeightMatrix(unsigned layer);

  private:
    static char weightAddr[50];
    zmq::context_t wctx;
    zmq::socket_t *dataSocket;
    zmq::socket_t *weightSocket;
    zmq::message_t confirm;
    unsigned nodeId;
    unsigned wPort;
    bool wsocktReady;

    std::vector<Matrix *> weights;
    std::thread wReqThread;
    std::thread wSndThread;
    std::thread infoThread;
};


class CPUComm : public ResourceComm {

  public:

    CPUComm(unsigned nodeId_, unsigned numNodes_, unsigned dataserverPort_, const std::string &wServersFile, unsigned wPort_, unsigned totalLayers_);

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
    zmq::socket_t weightSocket;
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

    std::vector<char *> weightServerAddrs;
    std::vector<Matrix *> weights;
    MessageServiceCPU msgService;
};


#endif // CPU_COMM_HPP
