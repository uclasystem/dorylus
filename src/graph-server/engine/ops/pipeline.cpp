#include "../engine.hpp"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
// Scheduler for sync/async pipeline
void Engine::scheduleAsyncFunc(unsigned tid) {
    double asyncStt = 0, asyncEnd = 0;
    double syncStt  = 0, syncEnd  = 0;
    // unsigned numAsyncEpochs = 0;
    const bool BLOCK = true;
    bool block = BLOCK;

    BackoffSleeper bs;
    while (!pipelineHalt) {
        schQueue.lock();
        if (schQueue.empty()) {
            schQueue.unlock();
            bs.sleep();
            continue;
        }

        Chunk c = schQueue.top();
        // printLog(nodeId, "SCHEDULER: Got %s", c.str().c_str());
        if (c.epoch > currEpoch) { // some chunk finishes curr epoch
            // (1) get a chunk for `numE + 1` means [1, numEpochs] finished
            if (c.epoch > numEpochs ||
                convergeState == CONVERGE_STATE::DONE) {
                unsigned finishedChunks = schQueue.size();
                schQueue.unlock();
                // (1.1) wait all chunks to finish
                if (finishedChunks < numLambdasForward) {
                    bs.sleep();
                    continue;
                } else { // (1.2) all chunks are done, exiting
                    if (async) {
                        asyncEnd = getTimer();
                    } else {
                        syncEnd = getTimer();
                        double epochTime = syncEnd - syncStt;
                        addEpochTime(epochTime);
                        printLog(nodeId, "Time for epoch %u: %.2lfms",
                                 currEpoch, epochTime);
                    }
                    if (numAsyncEpochs) {
                        double totalAsyncTime = asyncEnd - asyncStt;
                        asyncAvgEpochTime = totalAsyncTime / numAsyncEpochs;
                    }
                    pipelineHalt = true;
                    break;
                }
            }
            // (2) converge state switches so we turn off async pipeline
            if (async && convergeState != CONVERGE_STATE::EARLY) {
                if (maxEpoch == 0) { // haven't synced max epoch
                    schQueue.unlock();
                    // (2.1) master thread sync epoch with other nodes
                    if (tid == 0) {
                        maxEpoch = nodeManager.syncCurrEpoch(currEpoch);
                        // printLog(nodeId, "Max epoch %u", maxEpoch);
                    } else { // block other threads if any
                        bs.sleep();
                    }
                    continue;
                }
                // (2.2) wait all chunks finishing maxEpoch
                if (c.epoch > maxEpoch) {
                    unsigned finishedChunks = schQueue.size();
                    schQueue.unlock();
                    if (finishedChunks < numLambdasForward) {
                        bs.sleep();
                        continue;
                    } else { // (2.3) all chunks finish, switch to sync
                        nodeManager.barrier();
                        // nodeManager.readEpochUpdates();
                        printLog(nodeId, "Switch to sync from %u",
                                 maxEpoch + 1);
                        // reset scatter status
                        recvCnt = 0;
                        ghostVtcsRecvd = 0;
                        // reset [min|max] epoch info
                        minEpoch = maxEpoch + 1;
                        maxEpoch = 0;

                        async = false;
                        // reset timer
                        asyncEnd = getTimer();
                        syncStt = asyncEnd;
                        // reset block status
                        block = BLOCK;
                        continue;
                    }
                }
            }
            // (3) Bounded-staleness
            if (async && c.epoch > minEpoch + staleness) {
                schQueue.unlock();
                nodeManager.readEpochUpdates();
                // block until minEpoch being updated
                if (c.epoch > minEpoch + staleness)
                    bs.sleep();
                continue;
            }
            // (4) Sync mode
            if (!async && block) {
                // (4.1) Sync all chunks after a epoch
                if (tid == 0 && schQueue.size() == numLambdasForward) {
                    schQueue.unlock();
                    // Only master thread will call barrier
                    nodeManager.barrier();
                    block = false;
                } else { // Waiting all chunks finish or not master thd
                    schQueue.unlock();
                    bs.sleep();
                }
                continue;
            }
            // (4.2) Timing, skip epoch 0 and the async-sync transition epoch
            if (!async && currEpoch >= minEpoch) {
                syncEnd = getTimer();
                double epochTime = syncEnd - syncStt;
                addEpochTime(epochTime);
                printLog(nodeId, "Time for epoch %u: %.2lfms",
                    currEpoch, epochTime);
                syncStt = syncEnd;
                // reset block status
                block = BLOCK;
            }

            // (5) Start of the pipeline
            // (5.1) Epoch 0. Training begining
            if (currEpoch == START_EPOCH) {
                layer = 0;
                maxEpoch = 0;
                async = false;
                syncStt = getTimer();
                block = BLOCK;
            // (5.2) Epoch 1. Async pipeline starts
            } else if (currEpoch == START_EPOCH + 1) {
                ++minEpoch; // assert(minEpoch == 1)
                async = mode == LAMBDA &&
                        pipeline &&
                        staleness != UINT_MAX;
                if (async) {
                    printLog(nodeId, "Switch to async at epoch %u",
                            currEpoch);
                    asyncStt = getTimer();
                }
            }

            if (async)
                ++numAsyncEpochs;
            else
                ++numSyncEpochs;
            ++currEpoch;
            schQueue.pop();
            schQueue.unlock();

            // some initialization...
            layer = 0;
            if (async)
                printLog(nodeId, "Async Epoch %u [%u:%u] starts...",
                         currEpoch, minEpoch, minEpoch + staleness);
            else
                printLog(nodeId, "Sync Epoch %u starts...", currEpoch);
        } else {
            schQueue.pop();
            schQueue.unlock();
        }

        if (gnn_type == GNN::GCN) {
            GAQueue.push_atomic(c);
        } else if (gnn_type == GNN::GAT) {
            AVQueue.push_atomic(c);
        } else {
            abort();
        }

        bs.reset();
    }
    schQueue.clear();
}
#pragma GCC diagnostic pop

void Engine::gatherWorkFunc(unsigned tid) {
    BackoffSleeper bs;
    while (!pipelineHalt) {
        GAQueue.lock();
        if (GAQueue.empty()) {
            GAQueue.unlock();
            bs.sleep();
            continue;
        }

        Chunk c = GAQueue.top();
        // printLog(nodeId, "GA: Got %s", c.str().c_str());
        GAQueue.pop();
        GAQueue.unlock();

        if (gnn_type == GNN::GCN) {
            aggregateGCN(c);
            // applyVertexGCN(c);
            AVQueue.push_atomic(c);
        } else if (gnn_type == GNN::GAT) {
            aggregateGAT(c);
            if (c.dir == PROP_TYPE::FORWARD &&
                c.layer == numLayers) { // last forward layer
                predictGAT(c);

                c.dir = PROP_TYPE::BACKWARD; // switch direction
                SCQueue.push_atomic(c);
            } else {
                AVQueue.push_atomic(c);
            }
        } else {
            abort();
        }

        bs.reset();
    }
    GAQueue.clear();
}

// We could merge GA and AV since GA always calls AV
void Engine::applyVertexWorkFunc(unsigned tid) {
    BackoffSleeper bs;
    while (!pipelineHalt) {
        AVQueue.lock();
        if (AVQueue.empty()) {
            AVQueue.unlock();
            bs.sleep();
            continue;
        }

        Chunk c = AVQueue.top();
        c.vertex = true;
        // Note: here the chunk layer may be wrong for AVB, because AVB has a
        // pre-barrier inside applyVertex[GCN|GAT] to update the chunk layer.
        // printLog(nodeId, "AV: Got %s", c.str().c_str());
        AVQueue.pop();
        AVQueue.unlock();

        if (gnn_type == GNN::GCN)
            applyVertexGCN(c);
        else if (gnn_type == GNN::GAT)
            applyVertexGAT(c);
        else
            abort();

        bs.reset();
    }
    AVQueue.clear();
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
void Engine::scatterWorkFunc(unsigned tid) {
    BackoffSleeper bs;
    const bool BLOCK = true;
    bool block = BLOCK;
    while (!pipelineHalt) {
        // Sync all nodes during scatter
        if (SCStashQueue.size() == numLambdasForward) {
            if (tid == 0) {
                unsigned totalGhostCnt = currDir == PROP_TYPE::FORWARD
                                       ? graph.srcGhostCnt
                                       : graph.dstGhostCnt;
                recvCntLock.lock();
                while (recvCnt > 0 || ghostVtcsRecvd != totalGhostCnt) {
                    recvCntCond.wait();
                    // usleep(1000 * 1000);
                }
                recvCntLock.unlock();
                nodeManager.barrier();
                block = BLOCK;
                recvCnt = 0;
                ghostVtcsRecvd = 0;
                while (!SCStashQueue.empty()) {
                    Chunk sc = SCStashQueue.top();
                    SCStashQueue.pop();
                    AEQueue.push_atomic(sc);
                }
            } else {
                bs.sleep();
                continue; // other threads wait on SCQueue
            }
        }

        SCQueue.lock();
        if (SCQueue.empty()) {
            SCQueue.unlock();
            bs.sleep();
            continue;
        }
#if defined(_CPU_ENABLED_) || defined(_GPU_ENABLED_)
        // A barrier for pipeline
        // This barrier is for CPU/GPU only at the beginning of
        // the scatter phase to prevent someone send messages
        // too early.
        else if (block) {
            if (tid == 0 && SCQueue.size() == numLambdasForward) {
                SCQueue.unlock();
                nodeManager.barrier();
                block = false;
                SCQueue.lock();
            } else {
                SCQueue.unlock();
                bs.sleep();
                continue;
            }
        }
#endif

        Chunk c = SCQueue.top();
        // printLog(nodeId, "SC: Got %s", c.str().c_str());
        unsigned absLayer = getAbsLayer(c);
        if (absLayer != layer) {
            layer = absLayer;
            currDir = c.dir;
        }
        SCQueue.pop();
        SCQueue.unlock();

        if (gnn_type == GNN::GCN) {
            scatterGCN(c);
        } else if (gnn_type == GNN::GAT) {
            scatterGAT(c);
        } else {
            abort();
        }

        // Sync-Scatter for sync-pipeline and
        // the first epoch in asyn-pipeline only
        if (!async || c.epoch == 1) {
            SCStashQueue.push_atomic(c);
        } else {
            AEQueue.push_atomic(c);
        }

        bs.reset();
    }
    SCQueue.clear();
}
#pragma GCC diagnostic pop

void Engine::ghostReceiverFunc(unsigned tid) {
    switch (gnn_type) {
        case GNN::GCN:
            ghostReceiverGCN(tid);
            break;
        case GNN::GAT:
            ghostReceiverGAT(tid);
            break;
        default:
            abort();
    }
}

// Only for single thread because of the barrier
void Engine::applyEdgeWorkFunc(unsigned tid) {
    BackoffSleeper bs;
    while (!pipelineHalt) {
        AEQueue.lock();
        if (AEQueue.empty()) {
            AEQueue.unlock();
            bs.sleep();
            continue;
        }

        Chunk c = AEQueue.top();
        c.vertex = false;
        // printLog(nodeId, "AE: Got %s", c.str().c_str());
        AEQueue.pop();
        AEQueue.unlock();

        if (gnn_type == GNN::GCN) {
            applyEdgeGCN(c); // do nothing but push chunk to GAQueue
        } else if (gnn_type == GNN::GAT) {
            applyEdgeGAT(c);
        } else {
            abort();
        }

        bs.reset();
    }
    AEQueue.clear();
}

// [Deprecated] Sync pipeline scheduler
void Engine::scheduleFunc(unsigned tid) {
    printLog(nodeId, "Using deprecated func %s", __PRETTY_FUNCTION__);
    double sttTime = getTimer();
    double endTime;

    BackoffSleeper bs;
    while (!pipelineHalt) {
        schQueue.lock();
        if (schQueue.empty()) {
            schQueue.unlock();
            bs.sleep();
            continue;
        }

        Chunk c = schQueue.top();
        // printLog(nodeId, "SCHEDULER: Got %s", c.str().c_str());
        if (c.epoch > currEpoch) { // some chunk finishes curr epoch
            // Block until all chunks in this epoch finish
            if (schQueue.size() < numLambdasForward) {
                schQueue.unlock();
                bs.sleep();
                continue;
            } else  { // Enter next epoch. This is an atomic section
                endTime = getTimer();
                if (currEpoch == 0) { // Epoch 0. Training begining
                    layer = 0;
                    async = mode == LAMBDA && staleness != UINT_MAX;
                    printLog(nodeId, "Async: %d", async);
                } else { // Timing, skip epoch 0
                    unsigned epochTime = endTime - sttTime;
                    addEpochTime(epochTime);
                    printLog(nodeId, "Time for epoch %u: %lfms",
                        currEpoch, endTime - sttTime);
                }
                sttTime = endTime;
                ++currEpoch;
                schQueue.pop();
                schQueue.unlock();

                nodeManager.barrier();
                // get a chunk for `numE + 1` means [1, numEpochs] finished
                if (currEpoch >= numEpochs + 1 ||
                    convergeState == CONVERGE_STATE::DONE) {
                    pipelineHalt = true;
                    break;
                }

                // some initialization...
                layer = 0;
                printLog(nodeId, "Epoch %u starts...", currEpoch);
            }
        } else {
            schQueue.pop();
            schQueue.unlock();
        }

        if (gnn_type == GNN::GCN) {
            GAQueue.lock();
            GAQueue.push(c);
            GAQueue.unlock();
        }

        bs.reset();
    }
    schQueue.clear();
}
