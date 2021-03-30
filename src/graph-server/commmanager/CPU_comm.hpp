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
    void prefetchWeights() { msgService.prefetchWeightsMatrix(); };

private:
    // compute related
    void vtxNNForward(unsigned layer, bool lastLayer);
    void vtxNNBackward(unsigned layer);
    void edgNNForward(unsigned layer, bool lastLayer);
    void edgNNBackward(unsigned layer);

    void getTrainStat(Matrix &preds, Matrix &labels, float &acc,
                           float &loss);
    void maskout(Matrix &preds, Matrix &labels);
    unsigned totalLayers;
    unsigned nodeId;
    unsigned numNodes;
    unsigned numLocalVertices;

    std::vector<TensorMap> &savedNNTensors;

    Engine *engine;
    GNN gnn_type;

    // ntw related objs
    unsigned dPort;
    unsigned wPort;
    std::string wServersFile;
    std::vector<char *> weightServerAddrs;
    MessageService msgService;

    // GCN specific
    void vtxNNForwardGCN(unsigned layer, bool lastLayer);
    void vtxNNBackwardGCN(unsigned layer);
    void edgNNForwardGCN(unsigned layer, bool lastLayer) {}
    void edgNNBackwardGCN(unsigned layer) {}
    // GAT specific
    void vtxNNForwardGAT(unsigned layer, bool lastLayer);
    void vtxNNBackwardGAT(unsigned layer);
    void edgNNForwardGAT(unsigned layer, bool lastLayer);
    void edgNNBackwardGAT(unsigned layer);
};

Matrix activateDerivative(Matrix &mat);
Matrix hadamardMul(Matrix &A, Matrix &B);
Matrix hadamardSub(Matrix &A, Matrix &B);
Matrix softmax(Matrix &mat);
Matrix activate(Matrix &mat);
// GAT compute utils
Matrix expandDot(Matrix &m, Matrix &v, CSCMatrix<EdgeType> &forwardAdj);
Matrix expandHadamardMul(Matrix &m, Matrix &v, CSCMatrix<EdgeType> &forwardAdj);
Matrix expandMulZZ(FeatType **edgFeats, unsigned edgCnt, unsigned featDim);
Matrix reduce(Matrix &mat);

Matrix leakyRelu(Matrix &mat);
Matrix leakyReluBackward(Matrix &mat);

void loadWeightServers(std::vector<char *> &addresses,
                       const std::string &wServersFile);

void deleteMatrix(Matrix &mat);
#endif  // CPU_COMM_HPP
