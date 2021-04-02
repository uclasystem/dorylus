#ifndef __COMP_SERVER_HPP__
#define __COMP_SERVER_HPP__

#include <time.h>

#include <boost/algorithm/string/trim.hpp>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "../commmanager/GPU_comm.hpp"
#include "../commmanager/message_service.hpp"
#include "../utils/utils.hpp"
#include "comp_unit.cuh"

unsigned nodeId;
class GPUComm;

class ComputingServer {
public:
    ComputingServer();
    ComputingServer(GPUComm *gpu_comm, GNN gnn_type_);

    // compute related
    void vtxNNForward(unsigned layer, bool lastLayer);
    void vtxNNBackward(unsigned layer);
    void edgNNForward(unsigned layer, bool lastLayer);
    void edgNNBackward(unsigned layer);

    void prefetchWeights() { msgService.prefetchWeightsMatrix(); };
private:
    GNN gnn_type;
    unsigned nodeId;
    unsigned totalLayers;
    std::vector<TensorMap> &savedNNTensors;

    GPUComm *gpuComm;
    ComputingUnit cu;

    std::vector<char *> weightServerAddrs;
    MessageService &msgService;

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

#endif
