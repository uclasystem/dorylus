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

    ServerWorker(zmq::context_t& ctx_, int32_t sock_type, int32_t nParts_, int32_t nextIterCols_, int32_t& counter_,
                 Matrix& matrix_, FeatType *zData_, FeatType *actData_)
        : matrix(matrix_), ctx(ctx_), nextIterCols(nextIterCols_), worker(ctx, sock_type),
          zData(zData_), actData(actData_), nParts(nParts_), count(counter_) {
        partCols = matrix.cols;
        partRows = std::ceil((float) matrix.rows / (float) nParts);
        offset = partRows * partCols;
        bufSize = offset * sizeof(FeatType);
    }

    // Continuously listens for incoming lambda connections and either sends
    // a partitioned matrix or receives computed results.
    void work();
    
private:

    // Partitions the data matrix according to the partition id and
    // send it to the lambda thread for computation.
    void sendMatrixChunk(zmq::socket_t& socket, zmq::message_t& client_id, int32_t partId);

    // Accepts an incoming connection from a lambda thread and receives
    // two matrices, a 'Z' matrix and the 'activations' matrix.
    void recvMatrixChunks(zmq::socket_t& socket, int32_t partId, int32_t rows, int32_t cols);

    Matrix& matrix;

    int32_t nextIterCols;
    FeatType *zData;
    FeatType *actData;

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

    LambdaComm() { }
    LambdaComm(std::string& nodeIp_, unsigned dataserverPort_, std::string& coordserverIp_, unsigned coordserverPort_,
               int32_t nParts_, int32_t numListeners_)
        : nodeIp(nodeIp_), dataserverPort(dataserverPort_),
          coordserverIp(coordserverIp_), coordserverPort(coordserverPort_),
          nParts(nParts_), numListeners(numListeners_),
          ctx(1), frontend(ctx, ZMQ_ROUTER), backend(ctx, ZMQ_DEALER) { }
    
    void startContext(FeatType *dataBuf_, int32_t rows_, int32_t cols_, int32_t nextIterCols_, unsigned layer_);
    void endContext();

    // Binds to a public port and a backend routing port for the 
    // worker threads to connect to. Spawns 'numListeners' number
    // of workers and connects the frontend socket to the backend
    // by proxy.
    void run();

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

	int32_t counter;

    unsigned layer;

	zmq::context_t ctx;
	zmq::socket_t frontend;
	zmq::socket_t backend;

	std::string& nodeIp;
	unsigned dataserverPort;

    std::string& coordserverIp;
    unsigned coordserverPort;
};


#endif // LAMBDA_COMM_HPP
