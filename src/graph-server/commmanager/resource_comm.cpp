#include "resource_comm.hpp"
#include "../engine/engine.hpp"

void ResourceComm::NNRecvCallback(Engine *engine, bool async, Chunk &chunk) {
    switch (engine->gnn_type) {
        case GNN::GCN:
            NNRecvCallbackGCN(engine, async, chunk);
            break;
        case GNN::GAT:
            NNRecvCallbackGAT(engine, async, chunk);
            break;
        default:
            abort();
    }
}

void ResourceComm::NNRecvCallbackGCN(Engine *engine, bool async, Chunk &chunk) {
    // If bounded-staleness enabled, increment the finished chunks for this epoch
    // If the min epoch has finished, allow chunks to move to the next epoch
    // TODO (JOHN): Consider using __sync_fetch ops here to make code cleaner and
    //  still achieve synchronization
    Chunk nextChunk = incLayer(chunk, engine->numLayers);
    if (isLastLayer(chunk)) {
        if (async) {
            if (engine->staleness != UINT_MAX) {
                unsigned ind = chunk.epoch % (engine->staleness + 1);
                engine->finishedChunkLock.lock();
                if (++(engine->numFinishedEpoch[ind]) == engine->numLambdasForward) {
                    engine->finishedChunkLock.unlock();
                    // printLog(nodeId, "FINISHED epoch %u. Total finished %u", chunk.epoch, engine->nodesFinishedEpoch[ind]+1);
                    engine->numFinishedEpoch[ind] = 0;

                    engine->sendEpochUpdate(chunk.epoch);
                    engine->finishedNodeLock.lock();
                    if (++(engine->nodesFinishedEpoch[ind]) == engine->numNodes + 1) {
                        ++(engine->minEpoch);
                        engine->nodesFinishedEpoch[ind] = 0;
                    }
                    engine->finishedNodeLock.unlock();
                } else {
                    engine->finishedChunkLock.unlock();
                }
            }
        }
        engine->schQueue.lock();
        engine->schQueue.push(nextChunk);
        engine->schQueue.unlock();
    } else {
        engine->SCQueue.lock();
        engine->SCQueue.push(nextChunk);
        engine->SCQueue.unlock();
    }
}

void ResourceComm::NNRecvCallbackGAT(Engine *engine, bool async, Chunk &chunk) {}