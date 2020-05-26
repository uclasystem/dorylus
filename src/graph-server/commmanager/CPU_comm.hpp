#ifndef __CPU_COMM_HPP__
#define __CPU_COMM_HPP__

#include <boost/algorithm/string/trim.hpp>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <zmq.hpp>

#include "../../common/matrix.hpp"
#include "../../common/utils.hpp"
#include "../engine/engine.hpp"
#include "../utils/utils.hpp"
#include "message_service.hpp"
#include "resource_comm.hpp"

class CPUComm : public ResourceComm {
   public:
    CPUComm(Engine *engine_);

    void setAsync(bool _async, unsigned currEpoch){};  // GPU always run synchronously
    unsigned getRelaunchCnt() { return 0u; };
    void NNCompute(Chunk &chunk);
    void NNSync(){};

    // compute related
    void vtxNNForward(unsigned layer, bool lastLayer);
    void vtxNNBackward(unsigned layer);

    void edgNNForward(unsigned layer, bool lastLayer);
    void edgNNBackward(unsigned layer);

    void getTrainStat(Matrix &preds, Matrix &labels, float &acc,
                           float &loss);
   private:
    unsigned totalLayers;
    unsigned nodeId;
    unsigned numNodes;
    unsigned numLocalVertices;

    unsigned currLayer;

    std::string wServersFile;

    // ntw related objs
    unsigned dPort;
    unsigned wPort;

    std::vector<TensorMap> &savedNNTensors;

    Engine *engine;

    std::vector<char *> weightServerAddrs;
    MessageService msgService;

    Chunk c;
};

Matrix activateDerivative(Matrix &mat);
Matrix hadamardSub(Matrix &A, Matrix &B);
Matrix softmax(Matrix &mat);
Matrix activate(Matrix &mat);
void loadWeightServers(std::vector<char *> &addresses,
                       const std::string &wServersFile);

void deleteMatrix(Matrix &mat);
#endif  // CPU_COMM_HPP
