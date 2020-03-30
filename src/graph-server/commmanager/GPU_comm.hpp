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

class ComputingServer;

/**
 *
 * Accelerate computation using GPU
 *
 */
class GPUComm : public ResourceComm {
  public:

    GPUComm(unsigned nodeId_, unsigned numNodes_, unsigned dataserverPort_, const std::string &wServersFile, unsigned wPort_, unsigned totalLayers_);

    void newContext(unsigned layer, Matrix &inputTensor_, Matrix &outputTensor_, std::vector<Matrix> *savedTensors_);
    void newContext(unsigned layer, Matrix &inputTensor_, Matrix &outputTensor_, Matrix &targetTensor_, std::vector<Matrix> *savedTensors_);

    // For forward-prop.
    void requestForward(unsigned layer, bool lastLayer);

    void applyVertexForward(unsigned layer, unsigned lambdaId, bool lastLayer) {};
    void applyEdgeForward(unsigned layer, unsigned lambdaId, bool lastLayer) {};
    void waitResForward(unsigned layer, bool lastLayer) {};

    // For backward-prop.
    void requestBackward(unsigned layer, bool lastLayer);

    void applyVertexBackward(unsigned layer, unsigned lambdaId, bool lastLayer) {};
    void applyEdgeBackward(unsigned layer, unsigned lambdaId, bool lastLayer) {};
    void waitResBackward(unsigned layer, bool lastLayer) {};

    //cannot be called if newContextBackward is never called due to the assignment of targetmatrix
    void sendTargetMatrix();

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


    Matrix &inputTensor;
    Matrix &outputTensor;
    Matrix &targetTensor;
    std::vector<Matrix> *savedTensors;

    ComputingServer *comp_server;
};


#endif // GPU_COMM_HPP
