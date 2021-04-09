#include "GPU_comm.hpp"

GPUComm::GPUComm(Engine *engine_)
    : engine(engine_), nodeId(engine_->nodeId), totalLayers(engine_->numLayers),
    wServersFile(engine_->weightserverIPFile), wPort(engine_->weightserverPort),
    numNodes(engine_->numNodes), savedNNTensors(engine_->savedNNTensors),
    msgService(wPort, nodeId, totalLayers, engine_->gnn_type),
    ngpus(engine_->ngpus)
    {
        comp_servers = std::vector<ComputingServer*>(ngpus);
        for (unsigned devId = 0; devId < ngpus; ++devId) {
            comp_servers[devId] = new ComputingServer(this, engine->gnn_type, engine->compUnits[devId]);
        }
    }

void GPUComm::NNCompute(Chunk &chunk) {
    unsigned layer = chunk.layer;
    ComputingServer* comp_server = comp_servers[chunk.localId];
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