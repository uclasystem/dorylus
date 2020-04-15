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

#include "versionmatrix.hpp"
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
    ServerWorker(zmq::context_t& ctx_, WeightServer& _ws,
                 std::vector< TensorMap >& updateStore_,
                 std::vector<VersionTensorMap>& _weightsStore,
                 unsigned& numLambdas_, unsigned& lambdaRecved_);

    ~ServerWorker();

    // Listens on lambda threads' request for weights.
    void work();

private:
    void sendWeights(zmq::message_t& client_id, unsigned layer, bool forward);
    void recvUpdates(zmq::message_t& client_id);
    void recvUpdate(zmq::message_t& client_id, unsigned layer);
    void setBackpropNumLambdas(zmq::message_t& client_id, unsigned numLambdas_);
    void terminateServer(zmq::message_t& client_id);

    void sendTensor(Matrix& tensor, unsigned& more);
    void sendTensors(zmq::message_t& client_id, Chunk &chunk);

    void recvUpdateTensor(Chunk &chunk, VersionTensorMap& weights);
    void recvTensors(zmq::message_t& client_id, Chunk &chunk);

    zmq::context_t &ctx;
    zmq::socket_t workersocket;

    std::vector<VersionTensorMap>& weightsStore;
    std::vector< TensorMap >& updateStore;

    unsigned& numLambdas;
    unsigned& lambdaRecved;

    // Reference back to weight server so we can tell it to average and apply
    // final weight gradients.
    WeightServer& ws;
};


#endif
