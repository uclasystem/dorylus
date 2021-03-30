#include "engine.hpp"
#include "../utils/utils.hpp"

#include <omp.h>

#include <algorithm>
#include <boost/algorithm/string/classification.hpp>  // Include boost::for is_any_of.
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/trim.hpp>
#include <boost/program_options.hpp>
#include <boost/program_options/parsers.hpp>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_set>

#include "../graph/dataloader.hpp"

#ifdef _GPU_ENABLED_
#include "../commmanager/GPU_comm.hpp"
extern CuMatrix *NormAdjMatrixIn;
extern CuMatrix *NormAdjMatrixOut;
extern CuMatrix *Norms;
extern ComputingUnit cu;
#endif
#ifdef _CPU_ENABLED_
#include "../commmanager/CPU_comm.hpp"
#endif
#ifdef _LAMBDA_ENABLED_
#include "../commmanager/lambda_comm.hpp"
#endif

/**
 *
 * Initialize the engine with the given command line arguments.
 *
 */
void Engine::init(int argc, char *argv[]) {
    printLog(404, "Engine starts initialization...");
    timeInit = -getTimer();

    parseArgs(argc, argv);

    // Initialize the node manager and communication manager.
    nodeManager.init(dshMachinesFile, myPrIpFile,
                     this);  // NodeManger should go first.

    nodeId = nodeManager.getMyNodeId();
    numNodes = nodeManager.getNumNodes();
    assert(numNodes <= 256);  // Cluster size limitation.
    outFile += std::to_string(nodeId);
    commManager.init(nodeManager);

    // Set number of layers and number of features in each layer. Also store the
    // prefix sum of config for offset querying use.
    readLayerConfigFile(layerConfigFile);
    numLayers = layerConfig.size() - 1;

    std::string graphFile =
        datasetDir + "graph." + std::to_string(nodeId) + ".bin";
    // detect whether preprocessed
    {
        std::ifstream gfile(graphFile.c_str(), std::ios::binary);
        if (!gfile.good() || forcePreprocess) {
            DataLoader dl(datasetDir, nodeId, numNodes, undirected);
            dl.preprocess();
        }
    }
    graph.init(graphFile);
    printGraphMetrics();

    for (unsigned i = 0; i < 2 * numLayers; i++) {
        vecTimeAggregate.push_back(0.0);
        vecTimeApplyVtx.push_back(0.0);
        vecTimeScatter.push_back(0.0);
        vecTimeApplyEdg.push_back(0.0);
        vecTimeLambdaInvoke.push_back(0.0);
        vecTimeLambdaWait.push_back(0.0);
    }

    // Save intermediate tensors during forward phase for backward computation.
    savedTensors = new std::vector<Matrix>[numLayers]; // YIFAN: remove this
    savedNNTensors.resize(numLayers + 1);
    savedEdgeTensors.resize(numLayers + 1);

    // Track the number of chunks finished at each epoch;
    if (staleness != UINT_MAX) {
        nodesFinishedEpoch.resize(staleness + 1);
        numFinishedEpoch.resize(staleness + 1);
    }

    // Init it here for collecting data when reading files
    forwardVerticesInitData = new FeatType[getFeatDim(0) * graph.localVtxCnt];
    forwardGhostInitData = new FeatType[getFeatDim(0) * graph.srcGhostCnt];
    // Create labels storage area. Read in labels and store as one-hot format.
    localVerticesLabels =
        new FeatType[layerConfig[numLayers] * graph.localVtxCnt];

    // Read in initial feature values (input features) & labels.
    readFeaturesFile(featuresFile);
    readLabelsFile(labelsFile);

#ifdef _GPU_ENABLED_
    printLog(nodeId, "Loading SparseMatrices for GPU");
    NormAdjMatrixIn = new CuMatrix();
    NormAdjMatrixOut = new CuMatrix();
    Matrix norms(graph.localVtxCnt, 1, graph.vtxDataVec.data());
    Norms = new CuMatrix(norms, cu.handle);
    CuMatrix::MemoryPool.erase(Norms->devPtr);
    NormAdjMatrixIn->loadSpCSC(cu.spHandle, graph);
    NormAdjMatrixOut->loadSpCSR(cu.spHandle, graph);
#endif

    // Initialize synchronization utilities.
    recvCnt = 0;
    recvCntLock.init();
    recvCntCond.init(recvCntLock);

    aggQueueLock.init();
    scatQueueLock.init();

    // Initialize computation thread barrier.
    barComp.init(cThreads);
    // Create computation workers thread pool.
    computePool = new ThreadPool(cThreads);
    computePool->createPool();
    printLog(nodeId, "Created %u computation threads.", cThreads);
    // Create data communicators thread pool.
    dataPool = new ThreadPool(dThreads);
    dataPool->createPool();
    printLog(nodeId, "Created %u data communicator threads.", dThreads);

    if (nodeId == 0) {
        weightComm = new WeightComm(weightserverIPFile, weightserverPort);
        weightComm->updateChunkCnt(
            numNodes *
            numLambdasForward);  // now set up weight servers only once
    } else {
        weightComm = NULL;
    }

#ifdef _LAMBDA_ENABLED_
    if (mode == LAMBDA) {  // Lambda
        resComm = new LambdaComm(this);
    }
#endif

#ifdef _CPU_ENABLED_
    if (mode == CPU) {  // CPU
        resComm = new CPUComm(this);
    }
#endif

#ifdef _GPU_ENABLED_
    if (mode == GPU) {  // GPU
        resComm = new GPUComm(this);
    }
#endif

    timeForwardProcess = 0.0;
    timeInit += getTimer();
    printLog(nodeId, "Engine initialization complete.");
}

/**
 *
 * Destroy the engine.
 *
 */
void Engine::destroy() {
    // printLog(nodeId, "Destroying the engine...");

    nodeManager.destroy();
    commManager.destroy();
    computePool->destroyPool();
    dataPool->destroyPool();

    recvCntLock.destroy();
    recvCntCond.destroy();

    if (nodeId == 0) {
        weightComm->shutdown();
        delete weightComm;
    }

    delete resComm;

    if (gnn_type == GNN::GCN) {
        delete[] forwardVerticesInitData;
        delete[] forwardGhostInitData;

        delete[] localVerticesLabels;
    } else if (gnn_type == GNN::GAT) {
        delete[] forwardGhostInitData;
        for (int i = 0; i < numLayers; i++) {
            auto &kkv = savedNNTensors[i];
            for (auto &kv : kkv) {
                if (kv.first == "A") {
                    continue;
                }
                kv.second.free();
            }
        }
        for (auto &kkv : savedEdgeTensors) {
            for (auto &kv : kkv) {
                delete[] kv.second;
            }
        }
    }
}


void Engine::preallocate_tensors(GNN gnn_type) {
    switch (gnn_type) {
        case GNN::GCN:
            preallocateGCN();
            break;
        case GNN::GAT:
            preallocateGAT();
            break;
        default:
            printLog(nodeId, "Unrecognized benchmark type");
    }
}


void Engine::run() {
    preallocate_tensors(gnn_type);
    start_time = getCurrentTime();
    switch (gnn_type) {
        case GNN::GCN:
        case GNN::GAT:
            runPipeline();  // GAT and GCN share the same pipeline
            break;
        default:
            printLog(nodeId, "Unsupported GNN type");
    }
    end_time = getCurrentTime();
}

void Engine::runPipeline() {
    using ThreadVector = std::vector<std::thread>;
    commHalt = false;
    pipelineHalt = false;
    unsigned commThdCnt = std::max(2u, cThreads / 4);
    // unsigned commThdCnt = cThreads;
    // unsigned commThdCnt = 1;

    auto gaWrkrFunc =
        std::bind(&Engine::gatherWorkFunc, this, std::placeholders::_1);
    ThreadVector gaWrkrThds;
    for (unsigned tid = 0; tid < cThreads; ++tid) {
        gaWrkrThds.push_back(std::thread(gaWrkrFunc, 2 + tid));
    }
    auto avWrkrFunc =
        std::bind(&Engine::applyVertexWorkFunc, this, std::placeholders::_1);
    ThreadVector avWrkrThds;
    for (unsigned tid = 0; tid < 1; ++tid) {
        avWrkrThds.push_back(std::thread(avWrkrFunc, tid));
    }
    auto scWrkrFunc =
        std::bind(&Engine::scatterWorkFunc, this, std::placeholders::_1);
    // std::thread swt(scWrkrFunc, 1);
    ThreadVector scWrkrThds;
    for (unsigned tid = 0; tid < commThdCnt; ++tid) {
        scWrkrThds.push_back(std::thread(scWrkrFunc, tid));
    }
    auto ghstRcvrFunc =
        std::bind(&Engine::ghostReceiverFunc, this, std::placeholders::_1);
    ThreadVector ghstRcvrThds;
    for (unsigned tid = 0; tid < commThdCnt; ++tid) {
        ghstRcvrThds.push_back(std::thread(ghstRcvrFunc, tid));
    }
    auto aeWrkrFunc =
        std::bind(&Engine::applyEdgeWorkFunc, this, std::placeholders::_1);
    ThreadVector aeWrkrThds;
    for (unsigned tid = 0; tid < 1; ++tid) {
        aeWrkrThds.push_back(std::thread(aeWrkrFunc, tid));
    }
    nodeManager.barrier();

    loadChunks();
    // Start scheduler
    auto schedFunc =
        std::bind(&Engine::scheduleAsyncFunc, this, std::placeholders::_1);
    ThreadVector schedulerThds;
    for (unsigned tid = 0; tid < 1; ++tid) {
        schedulerThds.push_back(std::thread(schedFunc, tid));
    }

    for (unsigned tid = 0; tid < 1; ++tid) {
        schedulerThds[tid].join();
    }
    // Wait for all nodes to finish
    nodeManager.barrier();
    commHalt = true;
    for (unsigned tid = 0; tid < cThreads; ++tid) {
        gaWrkrThds[tid].join();
    }
    for (unsigned tid = 0; tid < 1; ++tid) {
        avWrkrThds[tid].join();
        aeWrkrThds[tid].join();
    }
    for (unsigned tid = 0; tid < commThdCnt; ++tid)
        scWrkrThds[tid].join();
    for (unsigned tid = 0; tid < commThdCnt; ++tid)
        ghstRcvrThds[tid].join();

    {
        // clean up
        unsigned sender, topic;
        char *msgBuf = new char[MAX_MSG_SIZE];
        if (commManager.dataPullIn(&sender, &topic, msgBuf,
                                    MAX_MSG_SIZE)) {
            printLog(nodeId, "CLEAN UP: Still msgs in buffer");
        };
        while (commManager.dataPullIn(&sender, &topic, msgBuf,
                                    MAX_MSG_SIZE)) {};
        delete[] msgBuf;
    }
}

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

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
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
                nodeManager.readEpochUpdates();
                schQueue.unlock();
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

Engine engine;
