#ifndef __WEIGHT_SERVER_HPP__
#define __WEIGHT_SERVER_HPP__


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
#include "serverworker.hpp"


/**
 *
 * Class of the weightserver. Weightservers are only responsible for replying weight requests from lambdas,
 * and handle weight updates.
 * 
 */
class WeightServer {

public:

    WeightServer(unsigned _port, std::string& configFileName);

    // Runs the weightserver, start a bunch of worker threads and create a proxy through frontend to backend.
    void run();

    // Average and apply update batches.
    void applyUpdates();

private:

    // Read in layer configurations.
    void initializeWeightMatrices(std::string& configFileName);

    // Defines how many concurrent weightserver threads to use.
    enum { kMaxThreads = 5 };

    std::vector<unsigned> dims;
    std::vector<Matrix> weightMats;

    // List of Matrices for holding updates until they are
    // ready to be applied.
    std::vector<Matrix> updates;

    // Number of lambdas requests at backprop.
    unsigned numLambdas;
    unsigned count;

    zmq::context_t ctx;
    zmq::socket_t frontend;
    zmq::socket_t backend;
    unsigned port;
};


#endif
