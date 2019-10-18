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
#include "../../common/matrix.hpp"
#include "../../common/utils.hpp"


class LambdaComm;

/**
 *
 * Base class for a lambda communication worker.
 *
 */
class LambdaWorker {

public:

    LambdaWorker(LambdaComm *manager);

    ~LambdaWorker();

    // Continuously listens for incoming lambda connections.
    void work();

    // Used at context creation / destruction.
    void refreshState(Matrix actMatrix_, FeatType *zData_, FeatType *actData_, unsigned numFeatsNext_, bool eval);
    void refreshState(std::vector<Matrix> zMatrices_, std::vector<Matrix> actMatrices_, Matrix targetMatrix_);
    void refreshState(Matrix oldGradMatrix_, Matrix newGradMatrix_, Matrix targetMatrix_, std::vector<Matrix> *savedTensors); // YIFAN for backward

protected:

    unsigned nodeId;

    zmq::context_t& ctx;
    zmq::socket_t workersocket;

    unsigned numLambdasForward;
    unsigned numLambdasBackward;

    unsigned& countForward;     // Counting up until all lambdas have returned.
    unsigned& countBackward;

    unsigned& numCorrectPredictions;
    float& totalLoss;
    unsigned& numValidationVertices;
    unsigned& evalPartitions;

    // Whether or not to evaluate this epoch
    bool evaluate;
    std::vector<bool>& trainPartitions;

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
    void sendChunk(Matrix &srcMat, zmq::message_t& client_id, unsigned partId, bool forward);
    void recvChunk(Matrix &dstMat, zmq::message_t& client_id, unsigned partId);

    // Partitions the label matrix given a partition id and
    // and send that partition to the lambda thread for validation
    void sendTargetMatrix(zmq::message_t& client_id, unsigned partId);

    // Receive the summed loss and total correct for this model
    void recvValidationResults(zmq::message_t& client_id, zmq::message_t& header);

    Matrix oldGradMatrix;
    Matrix newGradMatrix;
    Matrix targetMatrix;
    std::vector<Matrix> *savedTensors;

    void sendBackpropChunks(zmq::message_t& client_id, unsigned partId);

    // Accepts an incoming 'finished' message.
    void recvBackpropFinishMsg(zmq::message_t& client_id);

    std::vector<Matrix> zMatrices;      // Matrices to send.
    std::vector<Matrix> actMatrices;
};


#endif
