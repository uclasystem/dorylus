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


static const int32_t HEADER_SIZE = sizeof(int32_t) * 4;
enum OP { PUSH, PULL, REQ, RESP };


/**
 *
 * Serialization fuctions.
 * 
 */
template<class T>
static void
serialize(char *buf, int32_t offset, T val) {
    std::memcpy(buf + (offset * sizeof(T)), &val, sizeof(T));
}

template<class T>
static T
parse(const char *buf, int32_t offset) {
    T val;
    std::memcpy(&val, buf + (offset * sizeof(T)), sizeof(T));
    return val;
}

// ID represents either layer or data partition, depending on server responding.
static void
populateHeader(char *header, int32_t op, int32_t id, int32_t rows = 0, int32_t cols = 0) {
    serialize<int32_t>(header, 0, op);
    serialize<int32_t>(header, 1, id);
    serialize<int32_t>(header, 2, rows);
    serialize<int32_t>(header, 3, cols);
}


/**
 *
 * Struct for a matrix.
 * 
 */
struct Matrix {
    int32_t rows;
    int32_t cols;
    FeatType *data;

    Matrix() { rows = 0; cols = 0; }
    Matrix(int _rows, int _cols) { rows = _rows; cols = _cols; }
    Matrix(int _rows, int _cols, FeatType *_data) { rows = _rows; cols = _cols; data = _data; }
    Matrix(int _rows, int _cols, char *_data) { rows = _rows; cols = _cols; data = (FeatType *) _data; }

    FeatType *getData() const { return data; }
    size_t getDataSize() const { return rows * cols * sizeof(FeatType); }

    void setRows(int32_t _rows) { rows = _rows; }
    void setCols(int32_t _cols) { cols = _cols; }
    void setDims(int32_t _rows, int32_t _cols) { rows = _rows; cols = _cols; }
    void setData(FeatType *_data) { data = _data; }

    bool empty() { return rows == 0 || cols == 0; }

    std::string shape() { return "(" + std::to_string(rows) + "," + std::to_string(cols) + ")"; }
};


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

        // Create a proxy pipe that connects frontend to backend. This thread hangs throughout the life
        // of the engine.
        std::thread tproxy([&] {
            try {
                zmq::proxy(static_cast<void *>(frontend), static_cast<void *>(backend), nullptr);
            } catch (std::exception& ex) {
                std::cerr << ex.what() << std::endl;
            }

            for (int i = 0; i < numListeners; ++i) {    // Delete when context terminates.
                delete worker_threads[i];
                delete workers[i];
            }
        });
        tproxy.detach();
    }
    
    void startContext(FeatType *dataBuf_, int32_t rows_, int32_t cols_, int32_t nextIterCols_, unsigned layer_);
    void endContext();

    // Sends a request to the coordination server for a given
    // number of lambda threads.
    void requestLambdas();

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
