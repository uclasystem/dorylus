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
#include <unordered_map>

#include "../utils/utils.hpp"
#include "../parallel/lock.hpp"
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
    LambdaWorker(LambdaComm *manager_);
    ~LambdaWorker();

    // Continuously listens for incoming lambda connections.
    void work(unsigned wid);
    void sendTensors(zmq::message_t& client_id, Chunk &chunk);
    void sendEdgeTensor(zmq::message_t& client_id, Chunk& chunk);
    void sendEdgeInfo(zmq::message_t& client_id, Chunk& chunk);
    void recvTensors(zmq::message_t& client_id, Chunk &chunk);
    void recvETensors(zmq::message_t& client_id, Chunk &chunk);
    void recvEvalData(zmq::message_t& client_id, Chunk &chunk);
    void markFinish(zmq::message_t& client_id, Chunk &chunk);

    LambdaComm *manager;
    zmq::socket_t workersocket;
    unsigned wid;

    void sendTensor(Matrix &tensor, Chunk &chunk, unsigned &more);
    int recvTensor(Chunk &chunk);
    int recvETensor(Chunk& chunk);

    void sendEdgeTensorChunk(Matrix& eTensor, Chunk& chunk);
    void sendTensor(Chunk& chunk);

    void sendRefChunk(Matrix &srcMat, zmq::message_t& client_id, unsigned partId, bool forward);

    // timing info for profiling
    unsigned recvTS;
    unsigned sendTS;
    void setTSinHdr(void *hdr_buf) {
        sendTS = timestamp_ms();
        *((unsigned *)hdr_buf + 5) = recvTS;
        *((unsigned *)hdr_buf + 6) = sendTS;
    }
    void setTSinCfm(void *cfm_buf) {
        sendTS = timestamp_ms();
        *((unsigned *)cfm_buf + 1) = recvTS;
        *((unsigned *)cfm_buf + 2) = sendTS;
    }
};


#endif
