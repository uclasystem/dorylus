#ifndef __COMP_SERVER_HPP__
#define __COMP_SERVER_HPP__

#include <zmq.hpp>
#include <chrono>
#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <sstream>
#include <boost/algorithm/string/trim.hpp>
#include <vector>
#include <condition_variable>
#include "comp_unit.cuh"
#include "../utils/utils.hpp"


const float LEARNING_RATE=0.1;

/** Struct for wrapping over the returned matrices. */
typedef struct {
    std::vector<Matrix> zMatrices;          // Layer 1 -> last.
    std::vector<Matrix> actMatrices;        // Layer 0 -> last.
    Matrix targetMatrix;
} GraphData;


class ComputingServer {
public:
    ComputingServer(zmq::context_t& dctx,unsigned dPort_,const std::string& wServersFile,unsigned wPort_);

    //read weight file 
    void loadWeightServers(std::vector<char *>& addresses, const std::string& wServersFile);
    void terminateWeightServers();
    void sendShutdownMessage(zmq::socket_t& weightsocket);

    // Keep listening to computing requests
    void run();

    //For forward
    Matrix requestWeightsMatrix( unsigned layer);
    Matrix requestFeatsMatrix(unsigned rows,unsigned cols);
    void sendMatrices(Matrix& zResult, Matrix& actResult);
    void processForward(zmq::message_t &header);

    //For validation
    void evaluateModel(Matrix& activations);
    Matrix requestTargetMatrix();
    unsigned checkAccuracy(Matrix& predictions, Matrix& labels);
    float checkLoss(Matrix& preds, Matrix& labels);

    //For backward
    GraphData requestForwardMatrices(unsigned numLayers);
    std::vector<Matrix> requestWeightsMatrices(unsigned numLayers);
    void processBackward(zmq::message_t &header);
    void sendInfoMessage(zmq::socket_t& weightsocket, unsigned numLambdas);
    std::vector<Matrix> gradientComputation(GraphData& graphData, std::vector<Matrix>& weightsData);
    void sendWeightsUpdates(std::vector<Matrix> weightsUpdates);



private:
    //ntw related objs
    zmq::context_t dctx;
    zmq::context_t wctx;
    zmq::socket_t dataSocket;
    zmq::socket_t weightSocket;

    std::vector<char*> weightServerAddrs;
    unsigned dPort;
    unsigned wPort;
    unsigned nodeId;

    char ipc_addr[50];

    //GPU part
    ComputingUnit cu;
};




#endif 
