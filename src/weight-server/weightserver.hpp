#ifndef __WEIGHT_SERVER_HPP__
#define __WEIGHT_SERVER_HPP__

#include <algorithm>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cmath>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <map>

#include <zmq.hpp>
#include <boost/algorithm/string/trim.hpp>

#include "serverworker.hpp"
#include "AdamOptimizer.hpp"
#include "../common/matrix.hpp"
#include "../common/utils.hpp"


#define NUM_LISTENERS 2

enum CTRL_MSG { MASTERUP, WORKERUP, INITDONE, ACK };
const float LEARNING_RATE = 0.01;

/**
 *
 * Class of the weightserver. Weightservers are only responsible for replying weight requests from lambdas,
 * and handle weight updates.
 *
 */
class WeightServer {
  public:
    WeightServer(std::string &weightServersFile, std::string &myPrIpFile,
                 unsigned _listenerPort, std::string &configFileName,
                 unsigned _serverPort, std::string &tmpFileName);
    ~WeightServer();

    // Runs the weightserver, start a bunch of worker threads and create a proxy through frontend to backend.
    void run();

    // Average and apply update batches.
    void applyUpdate(unsigned layer, std::string& name);
    void synchronize(unsigned layer);

    //For sync
    std::mutex servers_updates_mutex;
    std::condition_variable servers_updates_cv;
    bool servers_updates_done;

  private:
    // For debugging.
    void serverLog(std::string info);

    // Use dsh file to open sockets to other weight servers for aggregation.
    void initializeWeightServerComms(std::string &weightServersFile, std::string &myPrIpFile);
    std::string parseNodeConfig(std::string &weightServersFile, std::string &myPrIpFile, std::string &myIp);

    // Read in layer configurations. The master constructs the weight matrices and then sends them
    // to the workers.
    void initializeWeightMatrices(std::string &configFileName);

    // Variations of weight matrix initialization
    Matrix xavierInitialization(unsigned dim1, unsigned dim2);
    Matrix kaimingInitialization(unsigned dim1, unsigned dim2);
    Matrix randomInitialization(unsigned dim1, unsigned dim2,
                                float lowerBound, float upperBound);

    // Initialize bias vectors
    Matrix initBias(unsigned dim, float initVal = 0);

    void initializeAdamVariables();

    void distributeWeightMatrices();

    std::vector<unsigned> dims;

    std::vector< TensorMap > weightsStore;
    std::vector< TensorMap > updateStore;
    std::vector<Matrix> biases;

    // Adam descent variables
    bool adam;  // whether to use standard SGD or Adam Opt

    // Number of lambdas requests at backprop.
    unsigned numLambdas;
    unsigned lambdaRecved;
    unsigned count;

    zmq::context_t ctx;
    zmq::socket_t frontend;
    zmq::socket_t backend;
    unsigned listenerPort;

    // Members related to communications.
    bool master;
    unsigned nodeId;
    std::vector<std::string> allNodeIps;

    zmq::context_t dataCtx;
    zmq::socket_t publisher;
    zmq::socket_t subscriber;
    unsigned serverPort;

    AdamOptimizer *adamOpt;
};


#endif
