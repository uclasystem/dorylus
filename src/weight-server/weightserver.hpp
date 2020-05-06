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
#include <mutex>

#include <zmq.hpp>
#include <boost/algorithm/string/trim.hpp>

#include "AdamOptimizer.hpp"
#include "weighttensor.hpp"
#include "../common/matrix.hpp"
#include "../common/utils.hpp"


#define NUM_LISTENERS 4

#define UPD_HEADER_SIZE (4 + sizeof(unsigned) * 2)
#define IDENTITY_SIZE 4
#define TENSOR_NAME_SIZE 8

enum CTRL_MSG { MASTERUP, WORKERUP, INITDONE, DATA, ACK };
const float LEARNING_RATE = 0.01;

class ServerWorker;

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
                 unsigned _serverPort, std::string &tmpFileName, bool _sync);
    ~WeightServer();

    bool sync; // sync mode or async pipeline

    void applyUpdate(unsigned layer, std::string& name);

    void receiver();
    std::thread *recvThd;
    std::mutex ackCntMtx;
    std::condition_variable ackCntCV;
    unsigned ackCnt;

    // Use dsh file to open sockets to other weight servers for aggregation.
    std::vector<std::string> parseNodeConfig(std::string &configFileName, std::string &weightServersFile, std::string &myPrIpFile);
    void initWServerComm(std::vector<std::string> &allNodeIps);
    // Members related to communications.
    unsigned nodeId;
    unsigned numNode;

    // Read in layer configurations. The master constructs the weight matrices and then sends them
    // to the workers.
    void initWeights();
    void distributeWeights();
    void freeWeights();
    std::vector<unsigned> dims;
    // [layer][name][version] -> versioned weight tensor
    std::vector<WeightTensorMap> weightsStore;
    std::vector<MutexMap> wMtxs;
    std::vector<MutexMap> uMtxs;
    std::mutex storeMtx;

    // Helper functions
    // Runs the weightserver, start a bunch of worker threads and create a proxy through frontend to backend.
    void run();
    void stopWorkers();
    std::vector<ServerWorker *> workers;
    std::vector<std::thread *> worker_threads;
    std::mutex termMtx;
    std::condition_variable termCV;
    bool term;

    void setLocalUpdTot(unsigned localUpdTot);
    void setGhostUpdTot(unsigned ghostUpdTot);
    unsigned numLambdas; // Number of update sent back from lambdas at backprop.

    void setupAdamOpt(bool adam);
    void deleteAdamOpt();
    bool adam;  // whether to use standard SGD or Adam Opt
    AdamOptimizer *adamOpt;

    void setupSockets();
    void closeSockets();
    void fillHeader(zmq::message_t &header, unsigned receiver, unsigned topic);
    void parseHeader(zmq::message_t &header, unsigned &sender, unsigned &topic);
    void fillTensorDescriber(zmq::message_t &td, std::string &name, unsigned layer);
    void parseTensorDescriber(zmq::message_t &td, std::string &name, unsigned &layer);
    void pushoutMsg(zmq::message_t &msg);
    void pushoutMsgs(std::vector<zmq::message_t *> &msgs);
    zmq::context_t ctx;
    zmq::socket_t frontend;
    zmq::socket_t backend;
    unsigned listenerPort;
    std::mutex pubMtx;
    std::mutex subMtx;
    zmq::context_t dataCtx;
    zmq::socket_t publisher;
    zmq::socket_t subscriber;
    unsigned serverPort;

    void createOutputFile(std::string &fileName);
    void closeOutputFile();
    std::ofstream outfile;

    // Initializer
    // Variations of weight matrix initialization
    Matrix xavierInitializer(unsigned dim1, unsigned dim2);
    Matrix kaimingInitializer(unsigned dim1, unsigned dim2);
    Matrix randomInitializer(unsigned dim1, unsigned dim2,
                            float lowerBound, float upperBound);
    // Initialize bias vectors
    Matrix initBias(unsigned dim, float initVal = 0);

    // For debugging.
    void serverLog(std::string info);
};

#endif
