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
#include <unistd.h>

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

    LambdaWorker(LambdaComm *manager_, FuncPtr _scatterFunc);

    ~LambdaWorker();

    // Continuously listens for incoming lambda connections.
    void work();

    // Used at context creation / destruction.
    void refreshState(Matrix actMatrix_, FeatType *zData_, FeatType *actData_,
      unsigned numFeatsNext_, bool _pipeline = false);
    void refreshState(Matrix oldGradMatrix_, Matrix newGradMatrix_,
      Matrix targetMatrix_, std::vector<Matrix> *savedTensors);

protected:
    LambdaComm *manager;

    zmq::socket_t workersocket;

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
    void fakeRecvChunks(zmq::message_t& client_id, unsigned chunkCnt);

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
    void recvChunk(Matrix &dstMat, zmq::message_t& client_id, unsigned partId, bool forward);

    // Partitions the label matrix given a partition id and
    // and send that partition to the lambda thread for validation
    void sendTargetMatrix(zmq::message_t& client_id, unsigned partId);

    // Receive the summed loss and total correct for this model
    void recvValidationResults(zmq::message_t& client_id, zmq::message_t& header);

    Matrix oldGradMatrix;
    Matrix newGradMatrix;
    Matrix targetMatrix;
    std::vector<Matrix> *savedTensors;

    // Callback when lambda results are returned
    bool pipeline;
    FuncPtr scatterFunc;
};


#endif
