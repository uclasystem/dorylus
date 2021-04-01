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
    // Increment the layer of finished chunks
    // printLog(engine->nodeId, "get chunk %s", chunk.str().c_str());
    if (engine->isLastLayer(chunk)) {
        if (engine->async) {
            unsigned ind = chunk.epoch % (engine->staleness + 1);
            engine->finishedChunkLock.lock();
            if (++(engine->numFinishedEpoch[ind]) == engine->numLambdasForward) {
                engine->numFinishedEpoch[ind] = 0;
                engine->finishedChunkLock.unlock();

                engine->nodeManager.readEpochUpdates();
                engine->sendEpochUpdate(chunk.epoch);
                // 0: all nodes finish the epoch and the counter has been reset
                unsigned finshedNodes = engine->nodesFinishedEpoch[ind] == 0
                                      ? engine->numNodes
                                      : engine->nodesFinishedEpoch[ind];
                printLog(engine->nodeId, "FINISHED epoch %u. Total finished %u",
                         chunk.epoch, finshedNodes);
            } else {
                engine->finishedChunkLock.unlock();
            }
        }
        // End of an epoch. inclayer to enter the next epoch
        Chunk nextChunk = engine->incLayerGCN(chunk);
        engine->schQueue.push_atomic(nextChunk);
    } else { // Not the last layer (AVB0)
        if (chunk.dir == PROP_TYPE::FORWARD) { // Forward, inc layer after AV computation
            Chunk nextChunk = engine->incLayerGCN(chunk);
            engine->SCQueue.push_atomic(nextChunk);
        } else { // Backward, inc layer has been done in the beginning of AVB
            engine->SCQueue.push_atomic(chunk);
        }
    }
}

void ResourceComm::NNRecvCallbackGAT(Engine *engine, Chunk &chunk) {
    if (!chunk.vertex) { // AE & AEB
        engine->GAQueue.push_atomic(chunk); // always push to GAQueue
        return;
    }
    // AV & AVB
    if (engine->isLastLayer(chunk)) {
        if (engine->async) {
            unsigned ind = chunk.epoch % (engine->staleness + 1);
            engine->finishedChunkLock.lock();
            if (++(engine->numFinishedEpoch[ind]) == engine->numLambdasForward) {
                engine->numFinishedEpoch[ind] = 0;
                engine->finishedChunkLock.unlock();

                engine->nodeManager.readEpochUpdates();
                engine->sendEpochUpdate(chunk.epoch);
                // 0: all nodes finish the epoch and the counter has been reset
                unsigned finshedNodes = engine->nodesFinishedEpoch[ind] == 0
                                      ? engine->numNodes
                                      : engine->nodesFinishedEpoch[ind];
                printLog(engine->nodeId, "FINISHED epoch %u. Total finished %u",
                         chunk.epoch, finshedNodes);
            } else {
                engine->finishedChunkLock.unlock();
            }
        }
        Chunk nextChunk = engine->incLayerGAT(chunk);
        // printLog(engine->nodeId, "Reach Last layer %s, next chunk %s", chunk.str().c_str(), nextChunk.str().c_str());
        engine->schQueue.push_atomic(nextChunk);
    } else { // Not the last layer (AVB0)
        if (chunk.dir == PROP_TYPE::FORWARD) { // Forward, inc layer after AV computation
            Chunk nextChunk = engine->incLayerGAT(chunk);
            engine->SCQueue.push_atomic(nextChunk);
        } else { // Backward, inc layer has been done in the beginning of AVB
            engine->SCQueue.push_atomic(chunk);
        }
    }
}