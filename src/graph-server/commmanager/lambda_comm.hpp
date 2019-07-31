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
 * Class of a data server worker.
 * 
 */
class ServerWorker {

public:

    ServerWorker(zmq::context_t& ctx_, int32_t sock_type, int32_t nParts_, int32_t& counter_,
                 Matrix& matrix_, unsigned nodeId_)
        : matrix(matrix_), ctx(ctx_), worker(ctx, sock_type),
          nParts(nParts_), count(counter_), nodeId(nodeId_) { }

    // Called at the start of a lambda communication context.
    void refreshState(FeatType *zData_, FeatType *actData_, int32_t nextIterCols_);

    // Continuously listens for incoming lambda connections and either sends
    // a partitioned matrix or receives computed results.
    void work();

private:

    // Partitions the data matrix according to the partition id and
    // send it to the lambda thread for computation.
    void sendMatrixChunk(zmq::socket_t& socket, zmq::message_t& client_id, int32_t partId);

    // Accepts an incoming connection from a lambda thread and receives
    // two matrices, a 'Z' matrix and the 'activations' matrix.
    void recvMatrixChunks(zmq::socket_t& socket, zmq::message_t& client_id, int32_t partId,
                          int32_t rows, int32_t cols);

    Matrix& matrix;

    int32_t nextIterCols;
    FeatType *zData;
    FeatType *actData;

    unsigned nodeId;

    zmq::context_t& ctx;
    zmq::socket_t worker;

    int32_t bufSize;
    int32_t partRows;
    int32_t partCols;
    int32_t nParts;
    int32_t offset;

    // Counting down until all lambdas have returned.
    static std::mutex count_mutex;
    int32_t& count;
};


/**
 *
 * Class of a lambda threads communication handler.
 * 
 */
class LambdaComm {

public:

    LambdaComm(std::string nodeIp_, unsigned dataserverPort_, std::string coordserverIp_, unsigned coordserverPort_,
               unsigned nodeId_, int32_t nParts_, int32_t numListeners_)
        : nodeIp(nodeIp_), dataserverPort(dataserverPort_),
          coordserverIp(coordserverIp_), coordserverPort(coordserverPort_),
          nodeId(nodeId_), nParts(nParts_), numListeners(numListeners_), counter(0),
          ctx(1), frontend(ctx, ZMQ_ROUTER), backend(ctx, ZMQ_DEALER), sendsocket(ctx, ZMQ_REQ) {
        
        char dhost_port[50];
        sprintf(dhost_port, "tcp://*:%u", dataserverPort);
        frontend.bind(dhost_port);
        backend.bind("inproc://backend");

        char chost_port[50];
        sprintf(chost_port, "tcp://%s:%u", coordserverIp.c_str(), coordserverPort);
        sendsocket.connect(chost_port);

        // Create numListeners workers and detach them.
        for (int i = 0; i < numListeners; ++i) {
            workers.push_back(new ServerWorker(ctx, ZMQ_DEALER, nParts, counter, matrix, nodeId));
            
            worker_threads.push_back(new std::thread(std::bind(&ServerWorker::work, workers[i])));
            worker_threads[i]->detach();
        }

        // Create a proxy pipe that connects frontend to backend. This thread hangs throughout the lifetime
        // of the engine.
        std::thread tproxy([&] {
            try {
                zmq::proxy(static_cast<void *>(frontend), static_cast<void *>(backend), nullptr);
            } catch (std::exception& ex) {
                printLog(nodeId, "ERROR: %s\n", ex.what());
            }

            for (int i = 0; i < numListeners; ++i) {    // Delete after context terminated.
                delete worker_threads[i];
                delete workers[i];
            }
        });
        tproxy.detach();
    }
    
    // Start / End a lambda communication context.
    void startContext(FeatType *dataBuf_, int32_t rows_, int32_t cols_, int32_t nextIterCols_, unsigned layer_);
    void endContext();

    // Send a request to the coordination server for a given number of lambda threads.
    void requestLambdas();

    // Send a message to the coordination server to shutdown
    void sendShutdownMessage();

    // Buffers for received results.
    FeatType *getZData() { return zData; }             // Z values.
    FeatType *getActivationData() { return actData; }  // After activation.

private:

	Matrix matrix;

	int32_t nextIterCols;
	FeatType *zData;
	FeatType *actData;
		
	int32_t nParts;
	int32_t numListeners;
    std::vector<ServerWorker *> workers;
    std::vector<std::thread *> worker_threads;

	int32_t counter;

    unsigned layer;

	zmq::context_t ctx;
	zmq::socket_t frontend;
	zmq::socket_t backend;
    zmq::socket_t sendsocket;

    unsigned nodeId;
	std::string nodeIp;
	unsigned dataserverPort;

    std::string coordserverIp;
    unsigned coordserverPort;
};


#endif // LAMBDA_COMM_HPP
