#ifndef __LAMBDA_COMM_HPP__
#define __LAMBDA_COMM_HPP__


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
#include <climits>
#include <string>
#include <thread>
#include <vector>
#include <zmq.hpp>
#include "../utils/utils.hpp"
#include "../../utils/utils.hpp"
#include "lambdaworker.hpp"


/**
 *
 * Class of a lambda threads communication handler.
 *
 */
class LambdaComm {

public:

    LambdaComm(std::string nodeIp_, unsigned dataserverPort_, std::string coordserverIp_, unsigned coordserverPort_, unsigned nodeId_,
               unsigned numLambdasForward_, unsigned numLambdasBackward_);
    ~LambdaComm();

    void setTrainValidationSplit(float trainPortion, unsigned numLocalVertices);

    // For forward-prop.
    void newContextForward(FeatType *dataBuf, FeatType *zData,
        FeatType *actData, unsigned numLocalVertices, unsigned numFeats,
        unsigned numFeatsNext, bool eval);

    void requestLambdasForward(unsigned layer);

    void invokeLambdaForward(unsigned layer, unsigned lambdaId);
    void waitLambdaForward();

    // For backward-prop.
    void newContextBackward(FeatType **zBufs, FeatType **actBufs, FeatType *targetBuf,
                            unsigned numLocalVertices, std::vector<unsigned> layerConfig);

    void requestLambdasBackward(unsigned numLayers_);

    // Send a message to the coordination server to shutdown.
    void sendShutdownMessage();

private:

    unsigned numLambdasForward;
    unsigned numLambdasBackward;

    bool evaluate;
    std::vector<bool> trainPartitions;

    unsigned numListeners;

    unsigned countForward;
    unsigned countBackward;

    unsigned numCorrectPredictions;
    float totalLoss;
    unsigned numValidationVertices;
    unsigned evalPartitions;

    zmq::context_t ctx;
    zmq::socket_t frontend;
    zmq::socket_t backend;
    zmq::socket_t coordsocket;

    unsigned nodeId;
    std::string nodeIp;
    unsigned dataserverPort;

    std::string coordserverIp;
    unsigned coordserverPort;
};


#endif // LAMBDA_COMM_HPP
