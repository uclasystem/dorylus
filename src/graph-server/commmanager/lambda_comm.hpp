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

    LambdaWorker(unsigned nodeId_, zmq::context_t *ctx_, unsigned numLambdas_, unsigned& count_)
        : nodeId(nodeId_), ctx(ctx_), worker(*ctx, ZMQ_DEALER), numLambdas(numLambdas_), count(count_) { }

protected:

    unsigned nodeId;

    zmq::context_t *ctx;
    zmq::socket_t worker;

    unsigned numLambdas;

    // Counting down until all lambdas have returned.
    unsigned& count;
};


/**
 *
 * Class of a forward-prop lambda communication worker.
 * 
 */
class LambdaWorkerForward : public LambdaWorker {

public:

    LambdaWorkerForward(unsigned nodeId_, zmq::context_t *ctx_, unsigned numLambdasForward_, unsigned& countForward_,
                        Matrix *actMatrix_, FeatType *zData_, FeatType *actData_, unsigned numFeatsNext_)
        : LambdaWorker(nodeId_, ctx_, numLambdasForward_, countForward_),
          actMatrix(actMatrix_), zData(zData_), actData(actData_), numFeatsNext(numFeatsNext_) { }

    // Continuously listens for incoming lambda connections and either sends
    // a partitioned matrix or receives computed results.
    void work();

private:

    // Partitions the data matrix according to the partition id and
    // send it to the lambda thread for computation.
    void sendMatrixChunk(zmq::socket_t& socket, zmq::message_t& client_id, unsigned partId);

    // Accepts an incoming connection from a lambda thread and receives
    // two matrices, a 'Z' matrix and a corresponding 'activations' matrix.
    void recvMatrixChunks(zmq::socket_t& socket, zmq::message_t& client_id, unsigned partId);

    Matrix *actMatrix;

    unsigned numFeatsNext;

    FeatType *zData;
    FeatType *actData;
};


/**
 *
 * Class of a forward-prop lambda communication worker.
 * 
 */
class LambdaWorkerBackward : public LambdaWorker {

public:

    LambdaWorkerBackward(unsigned nodeId_, zmq::context_t *ctx_, unsigned numLambdasBackward_, unsigned& countBackward_)
        : LambdaWorker(nodeId_, ctx_, numLambdasBackward_, countBackward_) { }

    // Continuously listens for incoming lambda connections and either sends
    // a partitioned matrix or receives computed results.
    void work();

private:

    // Partitions the data matrix according to the partition id and
    // send it to the lambda thread for computation.
    void sendMatrixChunks(zmq::socket_t& socket, zmq::message_t& client_id, unsigned partId);
};


/**
 *
 * Class of a lambda threads communication handler.
 * 
 */
class LambdaComm {

public:

    LambdaComm(std::string nodeIp_, unsigned dataserverPort_, std::string coordserverIp_, unsigned coordserverPort_, unsigned nodeId_,
               unsigned numLambdasForward_, unsigned numLambdasBackward_)
        : nodeIp(nodeIp_), dataserverPort(dataserverPort_), coordserverIp(coordserverIp_), coordserverPort(coordserverPort_),
          nodeId(nodeId_), numLambdasForward(numLambdasForward_), numLambdasBackward(numLambdasBackward_),
          countForward(0), countBackward(0) { }
    
    // For forward-prop.
    void newContextForward(FeatType *dataBuf, FeatType *zData, FeatType *actData, unsigned numLocalVertices,
                           unsigned numFeats, unsigned numFeatsNext);
    void requestLambdasForward(unsigned layer);
    void endContextForward();

    // For backward-prop.
    void newContextBackward();
    void requestLambdasBackward();
    void endContextBackward();

    // Send a message to the coordination server to shutdown.
    void sendShutdownMessage();

private:

    unsigned numLambdasForward;
    unsigned numLambdasBackward;

    unsigned numListeners;
    
    std::vector<LambdaWorkerForward *> forwardWorkers;
    std::vector<std::thread *> forwardWorker_threads;
    std::vector<LambdaWorkerBackward *> backwardWorkers;
    std::vector<std::thread *> backwardWorker_threads;

    unsigned countForward;
    unsigned countBackward;

    zmq::context_t *ctx;
    zmq::socket_t *frontend;
    zmq::socket_t *backend;
    zmq::socket_t *coordsocket;

    unsigned nodeId;
    std::string nodeIp;
    unsigned dataserverPort;

    std::string coordserverIp;
    unsigned coordserverPort;
};


#endif // LAMBDA_COMM_HPP
