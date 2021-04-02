#include "GPU_comm.hpp"

GPUComm::GPUComm(Engine *engine_)
    : engine(engine_), nodeId(engine_->nodeId), totalLayers(engine_->numLayers),
    wServersFile(engine_->weightserverIPFile), wPort(engine_->weightserverPort),
    numNodes(engine_->numNodes), savedNNTensors(engine_->savedNNTensors),
    msgService(wPort, nodeId, totalLayers, engine_->gnn_type) {
        comp_server = new ComputingServer(this, engine_->gnn_type);
    }

void GPUComm::NNCompute(Chunk &chunk) {
    unsigned layer = chunk.layer;
    if (chunk.vertex) { // AV & AVB
        if (chunk.dir == PROP_TYPE::FORWARD) {
            // printLog(nodeId, "GPU FORWARD vtx NN started");
            comp_server->vtxNNForward(layer, layer == (totalLayers - 1));
        }
        if (chunk.dir == PROP_TYPE::BACKWARD) {
            // printLog(nodeId, "GPU BACKWARD vtx NN started");
            comp_server->vtxNNBackward(layer);
        }
    } else { // AE & AEB
        layer--; // YIFAN: fix this
        if (chunk.dir == PROP_TYPE::FORWARD) {
            // printLog(nodeId, "GPU FORWARD edg NN started");
            comp_server->edgNNForward(layer, layer == (totalLayers - 1));
        }
        if (chunk.dir == PROP_TYPE::BACKWARD) {
            // printLog(nodeId, "GPU BACKWARD edg NN started");
            comp_server->edgNNBackward(layer);
        }
    }
    // printLog(nodeId, "GPU NN Done");
    NNRecvCallback(engine, chunk);
}

GPUComm::~GPUComm() {}