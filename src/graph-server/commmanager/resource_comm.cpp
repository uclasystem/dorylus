#include "resource_comm.hpp"
#include "../engine/engine.hpp"

void ResourceComm::NNRecvCallback(Engine *engine, Chunk &chunk) {
    switch (engine->gnn_type) {
        case GNN::GCN:
            NNRecvCallbackGCN(engine, chunk);
            break;
        case GNN::GAT:
            NNRecvCallbackGAT(engine, chunk);
            break;
        default:
            abort();
    }
}

void ResourceComm::NNRecvCallbackGCN(Engine *engine, Chunk &chunk) {
    // If bounded-staleness enabled, increment the finished chunks for this epoch
    Chunk nextChunk = incLayer(chunk, engine->numLayers);
    if (isLastLayer(chunk)) {
        if (engine->async) {
            unsigned ind = chunk.epoch % (engine->staleness + 1);
            engine->finishedChunkLock.lock();
            if (++(engine->numFinishedEpoch[ind]) == engine->numLambdasForward) {
                printLog(engine->nodeId, "FINISHED epoch %u. Total finished %u", chunk.epoch, engine->nodesFinishedEpoch[ind]+1);
                engine->numFinishedEpoch[ind] = 0;
                engine->finishedChunkLock.unlock();

                engine->sendEpochUpdate(chunk.epoch);
                engine->finishedNodeLock.unlock();
            } else {
                engine->finishedChunkLock.unlock();
            }
        }
        engine->schQueue.push_atomic(nextChunk);
    } else {
        engine->SCQueue.push_atomic(nextChunk);
    }
}

void ResourceComm::NNRecvCallbackGAT(Engine *engine, Chunk &chunk) {}