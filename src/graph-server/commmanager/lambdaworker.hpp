#ifndef __LAMBDA_WORKER_HPP__
#define __LAMBDA_WORKER_HPP__


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


/**
 *
 * Base class for a lambda communication worker.
 * 
 */
class LambdaWorker {

public:

    LambdaWorker(unsigned nodeId_, zmq::context_t& ctx_,
                 unsigned numLambdasForward_, unsigned numLambdasBackward_,
                 unsigned& countForward_, unsigned& countBackward_);

    // Continuously listens for incoming lambda connections.
    void work();

    // Used at context creation / destruction.
    void refreshState(Matrix actMatrix_, FeatType *zData_, FeatType *actData_, unsigned numFeatsNext_);
    void refreshState(std::vector<Matrix> zMatrices_, std::vector<Matrix> actMatrices_, Matrix targetMatrix_);

protected:

    unsigned nodeId;

    zmq::context_t& ctx;
    zmq::socket_t workersocket;

    unsigned numLambdasForward;
    unsigned numLambdasBackward;

    unsigned& countForward;     // Counting up until all lambdas have returned.
    unsigned& countBackward;

private:

    //
    // Forward-prop stuff.
    //

    // Partitions the data matrix according to the partition id and
    // send that partition to the lambda thread for computation.
    void sendAggregatedChunk(zmq::message_t& client_id, unsigned partId);

    // Accepts an incoming connection from a lambda thread and receives
    // two matrices, a 'Z' matrix and a corresponding 'activations' matrix.
    void recvLambdaResults(zmq::message_t& client_id, unsigned partId);

    Matrix actMatrix;   // Current layer's feats.
    unsigned numFeatsNext;
    FeatType *zData;    // Places to store the results from lambda.
    FeatType *actData;

    //
    // Backward-prop stuff.
    //

    // Partitions the needed matrices according to the partition id and
    // send that partition to the lambda thread for computation.
    void sendBackpropChunks(zmq::message_t& client_id, unsigned partId);

    // Accepts an incoming 'finished' message.
    void recvBackpropFinishMsg(zmq::message_t& client_id);

    std::vector<Matrix> zMatrices;      // Matrices to send.
    std::vector<Matrix> actMatrices;
    Matrix targetMatrix;
};


#endif
