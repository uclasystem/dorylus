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
        if (chunk.vertex) {
            printLog(nodeId, "GPU FORWARD VTX NN started");
            comp_server->vtxNNForward(currLayer,
                                      currLayer == (totalLayers - 1));
        } else {
            printLog(nodeId, "GPU FORWARD NN EDGE started");
            comp_server->edgNNForward(currLayer,
                                      currLayer == (totalLayers - 1));
        }
    }
    if (chunk.dir == PROP_TYPE::BACKWARD) {
        if (chunk.vertex) {
            printLog(nodeId, "GPU BACKWARD VTX NN started");
            comp_server->vtxNNBackward(currLayer);
        } else {
            printLog(nodeId, "GPU BACKWARD NN EDGE started");
            comp_server->edgNNBackward(currLayer);
        }
    }
    printLog(nodeId, "GPU NN Done");
}

void GPUComm::sendShutdownMessage() {
    printLog(nodeId, "Send Shutdown Message\n");
    // Send kill message.
    comp_server->terminate();
}

GPUComm::~GPUComm() { sendShutdownMessage(); }

void GPUComm::prefetchWeights() { comp_server->prefetchWeights(); }
