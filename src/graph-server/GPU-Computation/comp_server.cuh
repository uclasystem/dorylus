#ifndef __COMP_SERVER_HPP__
#define __COMP_SERVER_HPP__



#include "../commmanager/GPU_comm.hpp"

#include <chrono>
#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <sstream>
#include <boost/algorithm/string/trim.hpp>
#include <vector>
#include "comp_unit.cuh"
#include "../utils/utils.hpp"
#include "message_service.hpp"
#include <time.h>


const float LEARNING_RATE=0.00005;
unsigned nodeId;
class GPUComm;

class ComputingServer {
public:
    ComputingServer(){};
    ComputingServer(GPUComm* gpu_comm);

    //For forward
    void processForward(unsigned layer, bool lastLayer);

    //For validation
    // void evaluateModel(Matrix& activations);

    //For backward
    void processBackward(unsigned layer, bool lastLayer);
    void gradLayer(unsigned layer);
    void gradLoss(unsigned layer);

    void terminate();

private:

    std::vector<char*> weightServerAddrs;
    MessageService msgService;
    ComputingUnit cu;
    GPUComm* gpuComm;
    unsigned totalLayers;


};



#endif 
