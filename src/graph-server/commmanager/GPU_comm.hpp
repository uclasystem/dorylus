#ifndef __GPU_COMM_HPP__
#define __GPU_COMM_HPP__

#include <chrono>
#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <zmq.hpp>

#include "../GPU-Computation/comp_server.cuh"
#include "../../common/matrix.hpp"
#include "../../common/utils.hpp"
#include "../engine/engine.hpp"
#include "../utils/utils.hpp"
#include "message_service.hpp"
#include "resource_comm.hpp"

class ComputingServer;

/**
 *
 * Accelerate computation using GPU
 *
 */
class GPUComm : public ResourceComm {
public:
    GPUComm(Engine *engine_);
    ~GPUComm();

    void NNCompute(Chunk &chunk);
    friend class ComputingServer;
private:
    unsigned totalLayers;
    unsigned nodeId;
    unsigned numNodes;
    unsigned numLocalVertices;
    unsigned currLayer;
    unsigned ngpus;

    Engine *engine;

    //ntw related objs
    unsigned dPort;
    unsigned wPort;
    std::string wServersFile;
    MessageService msgService;

    std::vector<TensorMap> &savedNNTensors;

    std::vector<ComputingServer*> comp_servers;
};

#endif // GPU_COMM_HPP
