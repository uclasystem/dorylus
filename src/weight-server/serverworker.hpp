#ifndef __SERVER_WORKER_HPP__
#define __SERVER_WORKER_HPP__


#include <algorithm>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <fstream>
#include <functional>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <mutex>
#include <zmq.hpp>
#include <boost/algorithm/string/trim.hpp>

#include "weighttensor.hpp"
#include "weightserver.hpp"
#include "../common/matrix.hpp"
#include "../common/utils.hpp"


class WeightServer;


/**
 *
 * Wrapper over a server worker thread.
 *
 */
class ServerWorker {
public:
    ServerWorker(zmq::context_t& ctx_, WeightServer& _ws, unsigned tid);
    ~ServerWorker();

    // Listens on lambda threads' request for weights.
    void work();

private:
    void sendTensor(Matrix& tensor, unsigned& more);
    void sendTensors(zmq::message_t& client_id, Chunk &chunk);

    void recvUpdateTensor(Chunk &chunk, WeightTensorMap& weights);
    void recvTensors(zmq::message_t& client_id, Chunk &chunk);

    void recvEvalData(zmq::message_t& client_id, Chunk &chunk);

    void setNumLambdas(zmq::message_t& client_id, unsigned numLambdas_);
    void terminateServer(zmq::message_t& client_id);

    unsigned tid;
    zmq::context_t &ctx;
    zmq::socket_t workersocket;

    // Reference back to weight server so we can tell it to average and apply
    // final weight gradients.
    WeightServer& ws;
};


#endif
