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
#include <zmq.hpp>
#include <boost/algorithm/string/trim.hpp>
#include "../utils/utils.hpp"
#include "weightserver.hpp"


class WeightServer;


/**
 *
 * Wrapper over a server worker thread.
 * 
 */
class ServerWorker {

public:

    ServerWorker(zmq::context_t& ctx_, unsigned& counter, WeightServer& _ws,
                 std::vector<Matrix>& weights_, std::vector<Matrix>& updates_, unsigned& numLambdas_);

    // Listens on lambda threads' request for weights.
    void work();

private:

    void sendWeightsForwardLayer(zmq::message_t& client_id, unsigned layer);
    void sendWeightsBackward(zmq::message_t& client_id);
    void recvUpdates(zmq::message_t& client_id);
    void setBackpropNumLambdas(unsigned numLambdas_);
    void terminateServer();

    zmq::context_t &ctx;
    zmq::socket_t workersocket;

    std::vector<Matrix>& weightMats;
    std::vector<Matrix>& updates;
    
    unsigned& numLambdas;
    unsigned& count;

    // Reference back to weight server so we can tell it to average and apply
    // final weight gradients.
    WeightServer& ws;
};


#endif
