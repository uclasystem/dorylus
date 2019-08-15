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

#include <cblas.h>

#include "weightserver.hpp"


class WeightServer;


/**
 *
 * Wrapper over a server worker thread.
 * 
 */
class ServerWorker {

public:

    ServerWorker(zmq::context_t& ctx_, int sock_type, uint32_t& counter,
             std::vector<Matrix>& _weights, std::vector<Matrix>& _updates,
             std::vector<uint32_t>& _numLambdas, WeightServer& _ws);

    // Listens on lambda threads' request for weights.
    void work();

private:

    void sendWeights(zmq::socket_t& socket, zmq::message_t& client_id, int32_t layer);
    void recvUpdates(zmq::socket_t& socket, zmq::message_t& client_id, int32_t layer, zmq::message_t& header);
    void updateBackpropIterationInfo(int32_t layer, zmq::message_t& header);
    void terminateServer(zmq::socket_t& socket, zmq::message_t& client_id);



    zmq::context_t &ctx;
    zmq::socket_t worker;
    std::vector<Matrix>& weight_list;
    std::vector<Matrix>& updates;
    
    std::vector<uint32_t>& numLambdas;
    uint32_t& count;

    // Reference back to weight server so we can tell it to average and apply
    // final weight gradients
    WeightServer& ws;
};


#endif
