#include "GPU_comm.hpp"

GPUComm::GPUComm(Engine *engine_) : ResourceComm() {
    engine = engine_;
    nodeId = engine->nodeId;
    totalLayers = engine->numLayers;
    wServersFile = engine->weightserverIPFile;
    wPort = engine->weightserverPort;
    numNodes = engine->numNodes;
    currLayer = 0;
    comp_server = new ComputingServer(this);
}

void GPUComm::NNCompute(Chunk &chunk) {
    currLayer = chunk.layer;
    tensorMap = &engine->savedNNTensors[chunk.layer];
    if (chunk.dir == PROP_TYPE::FORWARD) {
        printLog(nodeId, "GPU FORWARD NN started");
        comp_server->processForward(currLayer, currLayer == (totalLayers - 1));
    }
    if (chunk.dir == PROP_TYPE::BACKWARD) {
        printLog(nodeId, "GPU BACKWARD NN started");
        comp_server->processBackward(currLayer);
    }
    printLog(nodeId, "GPU NN Done");
}

void GPUComm::sendShutdownMessage() {
    printLog(nodeId, "Send Shutdown Message\n");
    // Send kill message.
    comp_server->terminate();
}

GPUComm::~GPUComm() { sendShutdownMessage(); }