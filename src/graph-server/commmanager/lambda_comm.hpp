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


/**
 *
 * Base class for a lambda communication worker.
 * 
 */
class LambdaWorker {

public:

    LambdaWorker(unsigned nodeId_, zmq::context_t& ctx_, unsigned numLambdasForward_, unsigned numLambdasBackward_,
                 unsigned& countForward_, unsigned& countBackward_)
        : nodeId(nodeId_), ctx(ctx_), workersocket(ctx, ZMQ_DEALER),
          numLambdasForward(numLambdasForward_), numLambdasBackward(numLambdasBackward_),
          countForward(countForward_), countBackward(countBackward_) {
        workersocket.connect("inproc://backend");
    }

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


/**
 *
 * Class of a lambda threads communication handler.
 * 
 */
class LambdaComm {

public:

    LambdaComm(std::string nodeIp_, unsigned dataserverPort_, std::string coordserverIp_, unsigned coordserverPort_, unsigned nodeId_,
               unsigned numLambdasForward_, unsigned numLambdasBackward_)
        : nodeIp(nodeIp_), dataserverPort(dataserverPort_), coordserverIp(coordserverIp_), coordserverPort(coordserverPort_), nodeId(nodeId_), 
          ctx(1), frontend(ctx, ZMQ_ROUTER), backend(ctx, ZMQ_DEALER), coordsocket(ctx, ZMQ_REQ),
          numLambdasForward(numLambdasForward_), numLambdasBackward(numLambdasBackward_), numListeners(numLambdasBackward_),   // TODO: Decide numListeners.
          countForward(0), countBackward(0) {

        // Bind the proxy sockets.
        char dhost_port[50];
        sprintf(dhost_port, "tcp://*:%u", dataserverPort);
        frontend.bind(dhost_port);
        backend.bind("inproc://backend");

        char chost_port[50];
        sprintf(chost_port, "tcp://%s:%u", coordserverIp.c_str(), coordserverPort);
        coordsocket.connect(chost_port);

        // Create 'numListeners' workers and detach them.
        for (unsigned i = 0; i < numListeners; ++i) {
            workers.push_back(new LambdaWorker(nodeId, ctx, numLambdasForward, numLambdasBackward, countForward, countBackward));
            worker_threads.push_back(new std::thread(std::bind(&LambdaWorker::work, workers[i])));
            worker_threads[i]->detach();
        }

        // Create proxy pipes that connect frontend to backend. This thread hangs throughout the lifetime of this context.
        std::thread tproxy([&] {
            try {
                zmq::proxy(static_cast<void *>(frontend), static_cast<void *>(backend), nullptr);
            } catch (std::exception& ex) { /** Context termintated. */ }

            // Delete the workers after the context terminates.
            for (unsigned i = 0; i < numListeners; ++i) {
                delete workers[i];
                delete worker_threads[i];
            }
        });
        tproxy.detach();
    }
    
    // For forward-prop.
    void newContextForward(FeatType *dataBuf, FeatType *zData, FeatType *actData,
                           unsigned numLocalVertices, unsigned numFeats, unsigned numFeatsNext);
    void requestLambdasForward(unsigned layer);

    // For backward-prop.
    void newContextBackward(FeatType **zBufs, FeatType **actBufs, FeatType *targetBuf,
                            unsigned numLocalVertices, std::vector<unsigned> layerConfig);
    void requestLambdasBackward(unsigned numLayers_);

    // Send a message to the coordination server to shutdown.
    void sendShutdownMessage();

private:

    unsigned numLambdasForward;
    unsigned numLambdasBackward;

    unsigned numListeners;
    
    std::vector<LambdaWorker *> workers;
    std::vector<std::thread *> worker_threads;

    unsigned countForward;
    unsigned countBackward;

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
