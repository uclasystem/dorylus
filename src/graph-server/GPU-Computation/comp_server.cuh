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
    ComputingServer(GPUComm *gpu_comm);

    // GAT only
    void vtxNNForward(unsigned layer, bool lastLayer);
    void edgNNForward(unsigned layer, bool lastLayer);
    void edgNNBackward(unsigned layer);
    void vtxNNBackward(unsigned layer);

    // For forward
    void processForward(unsigned layer, bool lastLayer);

    // For validation
    // void evaluateModel(Matrix& activations);

    // For backward
    void processBackward(unsigned layer);
    void gradLayer(unsigned layer);
    void gradLoss(unsigned layer, CuMatrix pred, bool report = true);

    void prefetchWeights() { msgService.prefetchWeightsMatrix(totalLayers); };
    void terminate();

   private:
    std::vector<char *> weightServerAddrs;
    MessageService msgService;
    ComputingUnit cu;
    GPUComm *gpuComm;
    unsigned totalLayers;
    CuMatrix *weights;  // Weight Matrices Array
};

#endif
