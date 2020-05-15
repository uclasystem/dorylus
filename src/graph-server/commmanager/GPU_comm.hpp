#ifndef __GPU_COMM_HPP__
#define __GPU_COMM_HPP__


#include "../GPU-Computation/comp_server.cuh"
#include <chrono>
#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <zmq.hpp>
#include "resource_comm.hpp"
#include "../utils/utils.hpp"
#include "../../common/matrix.hpp"
#include "../../common/utils.hpp"
#include "../engine/engine.hpp"

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

    void setAsync(bool _async, unsigned currEpoch){};//GPU always run synchronously
    unsigned getRelaunchCnt() { return 0u; };
    void NNCompute(Chunk &chunk);
    void NNSync() {};

    void sendShutdownMessage();
    friend class ComputingServer;

  private:
    unsigned totalLayers;
    unsigned nodeId;
    unsigned numNodes;
    unsigned numLocalVertices;
    unsigned currLayer;

    std::string wServersFile;

    //ntw related objs
    unsigned dPort;
    unsigned wPort;

    TensorMap * tensorMap;

    ComputingServer *comp_server;
    Engine *engine;
    Chunk c;

};


#endif // GPU_COMM_HPP
