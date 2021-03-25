#include "engine.hpp"

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
    savedTensors = new std::vector<Matrix>[numLayers];
    savedNNTensors.resize(numLayers);
    savedEdgeTensors.resize(numLayers);

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

    preallocate_tensors(gnn_type);
    start_time = getCurrentTime();
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

void Engine::preallocateGAT() {
    unsigned vtxCnt = graph.localVtxCnt;

    // Store input tesnors
    savedNNTensors[0]["h"] =
        Matrix(vtxCnt, getFeatDim(0), forwardVerticesInitData);

//    savedNNTensors[0]["fg"] =
//        Matrix(graph.srcGhostCnt, getFeatDim(0), forwardGhostInitData);
    savedNNTensors[numLayers - 1]["lab"] =
        Matrix(vtxCnt, getFeatDim(numLayers), localVerticesLabels);

    // forward tensor allocation
    for (layer = 0; layer < numLayers; ++layer) {
        //unsigned featDim = getFeatDim(layer);
        unsigned nextFeatDim = getFeatDim(layer + 1);

        FeatType *zTensor = new FeatType[vtxCnt * nextFeatDim];
        std::memset(zTensor, 0, sizeof(FeatType) * vtxCnt * nextFeatDim);
        savedNNTensors[layer]["z"] = Matrix(vtxCnt, nextFeatDim, zTensor);

        // Technically not e_i because needs LeakyReLU
        FeatType* azTensor = new FeatType[graph.forwardAdj.nnz * 1];
        std::memset(azTensor, 0, sizeof(FeatType) * graph.forwardAdj.nnz * 1);
        savedNNTensors[layer]["az"] = Matrix(graph.forwardAdj.nnz, 1, azTensor);

        FeatType *ghostZTensor =
            new FeatType[graph.srcGhostCnt * nextFeatDim];
        std::memset(ghostZTensor, 0, sizeof(FeatType) * graph.srcGhostCnt * nextFeatDim);
        savedNNTensors[layer]["fg_z"] =
            Matrix(graph.srcGhostCnt, nextFeatDim, ghostZTensor);

        // Just storing these as matrices for easy access
        // Actually they are just edge values to be used with CSC/CSR
        FeatType* ATensor = graph.forwardAdj.values;
        std::memset(ATensor, 0, sizeof(FeatType) * graph.forwardAdj.nnz * 1);
        savedNNTensors[layer]["A"] =
            Matrix(graph.forwardAdj.nnz, 1, ATensor);

        // Attention scores stored in CSCMatrix<>::values

        FeatType **eVFeatsTensor = srcVFeats2eFeats(
            zTensor, ghostZTensor, vtxCnt, nextFeatDim);
        savedEdgeTensors[layer]["fedge"] = eVFeatsTensor;

        FeatType *ahTensor = new FeatType[vtxCnt * nextFeatDim];
        std::memset(ahTensor, 0, sizeof(FeatType) * vtxCnt * nextFeatDim);
        savedNNTensors[layer]["ah"] = Matrix("ah", vtxCnt, nextFeatDim, ahTensor);

        if (layer < numLayers - 1) {
            FeatType *hTensor = new FeatType[vtxCnt * nextFeatDim];
            std::memset(hTensor, 0, sizeof(FeatType) * vtxCnt * nextFeatDim);
            savedNNTensors[layer + 1]["h"] = Matrix(vtxCnt, nextFeatDim, hTensor);
        }
//            FeatType **edgeTensor =
//                srcVFeats2eFeats(hTensor, ghostTensor, vtxCnt, nextFeatDim);
//            savedEdgeTensors[layer + 1]["fedge"] = edgeTensor;
    }

    // backward tensor allocation
    for (layer = numLayers - 1; layer >= 0; --layer) {
        unsigned featDim = getFeatDim(layer + 1);

        // LOSS GRAD TENSORS
        FeatType *gradTensor = new FeatType[vtxCnt * featDim];
        savedNNTensors[layer]["grad"] =
            Matrix("grad", vtxCnt, featDim, gradTensor);

        // APPLY EDGE TENSORS
        FeatType* gradATensor =
            new FeatType[graph.forwardAdj.nnz * 1];
        std::memset(gradATensor, 0, sizeof(FeatType) * graph.forwardAdj.nnz * 1);
        savedNNTensors[layer]["dA"] =
            Matrix(graph.forwardAdj.nnz, 1, gradATensor);

        // GATHER TENSORS
        FeatType *aTgTensor = new FeatType[vtxCnt * featDim];
        savedNNTensors[layer]["aTg"] = Matrix(vtxCnt, featDim, aTgTensor);

        // SCATTER TENSORS
        FeatType *ghostTensor = new FeatType[graph.dstGhostCnt * featDim];
        savedNNTensors[layer]["bg_d"] =
            Matrix(graph.dstGhostCnt, featDim, ghostTensor);

        FeatType **eGrad =
            dstVFeats2eFeats(gradTensor, ghostTensor, vtxCnt, featDim);
        savedEdgeTensors[layer]["bedge"] = eGrad;
    }
}

void Engine::preallocateGCN() {
    unsigned vtxCnt = graph.localVtxCnt;

    // Store input tesnors
    savedNNTensors[0]["x"] =
        Matrix(vtxCnt, getFeatDim(0), forwardVerticesInitData);
    savedNNTensors[0]["fg"] =
        Matrix(graph.srcGhostCnt, getFeatDim(0), forwardGhostInitData);
    savedNNTensors[numLayers - 1]["lab"] =
        Matrix(vtxCnt, getFeatDim(numLayers), localVerticesLabels);

    FeatType **eVFeatsTensor = srcVFeats2eFeats(
        forwardVerticesInitData, forwardGhostInitData, vtxCnt, getFeatDim(0));
    savedEdgeTensors[0]["fedge"] = eVFeatsTensor;

    // forward tensor allocation
    for (layer = 0; layer < numLayers; ++layer) {
        unsigned featDim = getFeatDim(layer);
        unsigned nextFeatDim = getFeatDim(layer + 1);

        // GATHER TENSORS
        FeatType *ahTensor = new FeatType[vtxCnt * featDim];
        savedNNTensors[layer]["ah"] = Matrix("ah", vtxCnt, featDim, ahTensor);

        // APPLY TENSORS
        if (layer < numLayers - 1) {
            FeatType *zTensor = new FeatType[vtxCnt * nextFeatDim];
            FeatType *hTensor = new FeatType[vtxCnt * nextFeatDim];

            savedNNTensors[layer]["z"] = Matrix(vtxCnt, nextFeatDim, zTensor);
            savedNNTensors[layer]["h"] = Matrix(vtxCnt, nextFeatDim, hTensor);

            // SCATTER TENSORS
            FeatType *ghostTensor =
                new FeatType[graph.srcGhostCnt * nextFeatDim];
            savedNNTensors[layer + 1]["fg"] =
                Matrix(graph.srcGhostCnt, nextFeatDim, ghostTensor);

            FeatType **edgeTensor =
                srcVFeats2eFeats(hTensor, ghostTensor, vtxCnt, nextFeatDim);
            savedEdgeTensors[layer + 1]["fedge"] = edgeTensor;
        }
    }

    // backward tensor allocation
    for (layer = numLayers - 1; layer > 0; --layer) {
        unsigned featDim = getFeatDim(layer);

        // APPLY TENSORS
        FeatType *gradTensor = new FeatType[vtxCnt * featDim];
        savedNNTensors[layer]["grad"] =
            Matrix("grad", vtxCnt, featDim, gradTensor);

        // SCATTER TENSORS
        FeatType *ghostTensor = new FeatType[graph.dstGhostCnt * featDim];
        savedNNTensors[layer - 1]["bg"] =
            Matrix(graph.dstGhostCnt, featDim, ghostTensor);

        FeatType **eFeats =
            dstVFeats2eFeats(gradTensor, ghostTensor, vtxCnt, featDim);
        savedEdgeTensors[layer - 1]["bedge"] = eFeats;

        // GATHER TENSORS
        FeatType *aTgTensor = new FeatType[vtxCnt * featDim];
        savedNNTensors[layer - 1]["aTg"] = Matrix(vtxCnt, featDim, aTgTensor);
    }
}


void Engine::run() {
    switch (gnn_type) {
        case GNN::GCN:
            runGCN();
            break;
        case GNN::GAT:
            // runGAT();
            break;
        default:
            printLog(nodeId, "Unsupported GNN type");
    }
}

void Engine::runGCN() {
    using ThreadVector = std::vector<std::thread>;
    commHalt = false;
    pipelineHalt = false;
    unsigned commThdCnt = std::max(2u, cThreads / 4);
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
    for (unsigned tid = 0; tid < cThreads; ++tid) {
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
        std::bind(&Engine::ghostReceiver, this, std::placeholders::_1);
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

    loadChunksGCN();
    // Start scheduler
    auto schedFunc =
        std::bind(&Engine::scheduleFunc, this, std::placeholders::_1);
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
        avWrkrThds[tid].join();
    }
    for (unsigned tid = 0; tid < 1; ++tid)
        aeWrkrThds[tid].join();
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


/**
 *
 * Runs a forward propagation phase: (Aggregate -> Lambda Computing -> Ghost
 * Update) -> ( ... ) -> ... Will start a bunch of worker threads and a bunch of
 * data communicator threads.
 *
 */
FeatType *Engine::runForwardGCN(unsigned epoch) {
    currEpoch = epoch;
    // Make sure all nodes start running the forward-prop phase.
    if (nodeId == 0) {
        printLog(nodeId, "Epoch %u FORWARD starts...", epoch);
    }

    timeForwardProcess -= getTimer();

    // Create buffer for first-layer aggregation.
    FeatType *inputTensor = forwardVerticesInitData;
    forwardGhostVerticesDataIn = forwardGhostInitData;
    // FeatType **eVFeatsTensor = srcVFeats2eFeats(inputTensor,
    // forwardGhostInitData, graph.localVtxCnt, getFeatDim(layer));
    FeatType **eVFeatsTensor = savedEdgeTensors[0]["fedge"];
    for (layer = 0; layer < numLayers; ++layer) {
        inputTensor = aggregate(eVFeatsTensor, graph.localInEdgeCnt,
                                getFeatDim(layer), AGGREGATOR::WSUM);
        inputTensor =
            applyVertex(inputTensor, graph.localVtxCnt, getFeatDim(layer),
                        getFeatDim(layer + 1), layer == numLayers - 1);
        // inputTensor = fusedGatherApply(eVFeatsTensor, graph.localInEdgeCnt,
        //                 getFeatDim(layer), getFeatDim(layer + 1), AGGREGATOR::WSUM);
        if (layer < numLayers - 1) {  // don't need scatter at the last layer.
            eVFeatsTensor =
                scatter(inputTensor, graph.localVtxCnt, getFeatDim(layer + 1));
            eVFeatsTensor =
                applyEdge(NULL, graph.localInEdgeCnt, 0, eVFeatsTensor,
                          eVFeatsTensor + graph.localInEdgeCnt,
                          getFeatDim(layer + 1), getFeatDim(layer + 1));
        }
    }

    timeForwardProcess += getTimer();
    // if (nodeId == 0) {
    //     printLog(nodeId, "Epoch %u FORWARD finishes at layer %u.", epoch, layer);
    // }
    // calcAcc(inputTensor, localVerticesLabels, graph.localVtxCnt,
    // getFeatDim(numLayers));

    return inputTensor;
}

/**
 *
 * Runs a backward propagation phase: (Lambda Computing) -> ( ... ) -> ...
 * Will start a bunch of worker threads and a bunch of data communicator
 * threads.
 *
 */
void Engine::runBackwardGCN(FeatType *initGradTensor) {
    // if (nodeId == 0) {
    //     printLog(nodeId, "Epoch %u BACKWARD starts...", currEpoch);
    // }

    timeBackwardProcess -= getTimer();

    // Create buffer for first-layer aggregation.
    FeatType *gradTensor = initGradTensor;

    // Pure sequential
    FeatType **eVGradTensor = NULL;
    for (layer = numLayers - 1; layer > 0; --layer) {
        eVGradTensor =
            scatterBackward(gradTensor, graph.localVtxCnt, getFeatDim(layer));
        eVGradTensor = applyEdgeBackward(NULL, graph.localOutEdgeCnt, 0,
                                         eVGradTensor + graph.localOutEdgeCnt,
                                         eVGradTensor, getFeatDim(layer),
                                         getFeatDim(layer));
        gradTensor = aggregateBackward(eVGradTensor, graph.localOutEdgeCnt,
                                       getFeatDim(layer), AGGREGATOR::WSUM);
        gradTensor =
            applyVertexBackward(gradTensor, graph.localVtxCnt,
                                getFeatDim(layer - 1), getFeatDim(layer));
        // gradTensor = fusedGatherApplyBackward(eVGradTensor, graph.localOutEdgeCnt,
        //                 getFeatDim(layer - 1), getFeatDim(layer), AGGREGATOR::WSUM);
    }

    timeBackwardProcess += getTimer();
    // if (nodeId == 0) {
    //     printLog(nodeId, "Epoch %u BACKWARD finishes at layer %u.", currEpoch, layer);
    // }
    numSyncEpochs++;
}

/**
 * Run the deep-pipeline version where all stages happen in parallel
 */
void Engine::runAsyncPipelineGCN() {
    double stt = getTimer();

    resComm->setAsync(true, currEpoch);

    commHalt = false;
    maxEpoch = 0; // for async/sync switching

    unsigned commThdCnt = std::max(2u, cThreads / 4);
    // unsigned commThdCnt = 1;
    auto ghstRcvr =
        std::bind(&Engine::ghostReceiver, this, std::placeholders::_1);

    // std::thread grt(ghstRcvr, 0);
    std::vector<std::thread> ghstRcvrThds;
    for (unsigned tid = 0; tid < commThdCnt; ++tid) {
        ghstRcvrThds.push_back(std::thread(ghstRcvr, tid));
    }

    auto scttrWrkr =
        std::bind(&Engine::scatterWorker, this, std::placeholders::_1);
    // std::thread swt(scttrWrkr, 1);
    std::vector<std::thread> sctterWrkrThds;
    for (unsigned tid = 0; tid < commThdCnt; ++tid) {
        sctterWrkrThds.push_back(std::thread(scttrWrkr, tid));
    }

    auto aggWrkr =
        std::bind(&Engine::aggregator, this, std::placeholders::_1);
    std::vector<std::thread> aggWrkrThds;
    for (unsigned tid = 0; tid < cThreads; ++tid) {
        aggWrkrThds.push_back(std::thread(aggWrkr, 2 + tid));
    }
    for (unsigned tid = 0; tid < cThreads; ++tid) {
        aggWrkrThds[tid].join();
    }

    double edt = getTimer();
    asyncAvgEpochTime = (edt - stt) / numAsyncEpochs;

    // Wait for all nodes to finish
    nodeManager.barrier();
    commHalt = true;
    // swt.join();
    // grt.join();
    for (unsigned tid = 0; tid < commThdCnt; ++tid) {
        sctterWrkrThds[tid].join();
    }
    for (unsigned tid = 0; tid < commThdCnt; ++tid) {
        ghstRcvrThds[tid].join();
    }

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

    if (nodeId == 0) {
        printLog(nodeId, "All nodes finished pipeline");
    }

    resComm->setAsync(false, currEpoch - 1);
}

void Engine::runForwardSyncPipelineGCN(unsigned epoch) {
    currEpoch = epoch;
    // Make sure all nodes start running the forward-prop phase.
    if (nodeId == 0) {
        printLog(nodeId, "Epoch %u FORWARD starts...", epoch);
    }

    timeForwardProcess -= getTimer();

    for (layer = 0; layer < numLayers; ++layer) {
        fusedGAS();
    }

    timeForwardProcess += getTimer();
    // if (nodeId == 0) {
    //     printLog(nodeId, "Epoch %u FORWARD finishes at layer %u.", epoch, layer);
    // }
}

void Engine::runBackwardSyncPiplineGCN() {
    // if (nodeId == 0) {
    //     printLog(nodeId, "Epoch %u BACKWARD starts...", currEpoch);
    // }
    timeBackwardProcess -= getTimer();

    // Pure sequential, fused the first backward scatter with forward phase GAS
    FeatType **eVGradTensor = NULL;
    for (layer = numLayers - 1; layer > 0; --layer) {
        eVGradTensor = savedEdgeTensors[layer - 1]["bedge"];
        fusedGatherApplyBackward(eVGradTensor, graph.localOutEdgeCnt,
            getFeatDim(layer - 1), getFeatDim(layer), AGGREGATOR::WSUM);
    }

    timeBackwardProcess += getTimer();
    // if (nodeId == 0) {
    //     printLog(nodeId, "Epoch %u BACKWARD finishes at layer %u.", currEpoch, layer);
    // }
    numSyncEpochs++;
}

void Engine::aggregateGCN(Chunk &c) {
    unsigned lvid = c.lowBound;
    unsigned limit = c.upBound;

    if (c.dir == PROP_TYPE::FORWARD) { // forward
        unsigned featDim = getFeatDim(c.layer);
        FeatType *featTensor = NULL;
        if (c.layer == 0)
            featTensor =
                getVtxFeat(savedNNTensors[c.layer]["x"].getData(), lvid, featDim);
        else
            featTensor = getVtxFeat(savedNNTensors[c.layer - 1]["h"].getData(),
                                    lvid, featDim);

        FeatType **eFeatsTensor = savedEdgeTensors[c.layer]["fedge"];  // input tensor
        FeatType *aggTensor = savedNNTensors[c.layer]["ah"].getData(); // output tensor

        FeatType *chunkPtr = getVtxFeat(aggTensor, lvid, featDim);
        std::memcpy(chunkPtr, featTensor,
                    sizeof(FeatType) * (limit - lvid) * featDim);
        while (lvid < limit) {
            // Read out data of the current layer of given vertex.
            FeatType *currDataDst = getVtxFeat(aggTensor, lvid, featDim);
            // Apply normalization factor on the current data.
            {
                const EdgeType normFactor = graph.vtxDataVec[lvid];
                for (unsigned i = 0; i < featDim; ++i) {
                    currDataDst[i] *= normFactor;
                }
            }
            // Aggregate from incoming neighbors.
            for (uint64_t eid = graph.forwardAdj.columnPtrs[lvid];
                eid < graph.forwardAdj.columnPtrs[lvid + 1]; ++eid) {
                EdgeType normFactor = graph.forwardAdj.values[eid];
                for (unsigned j = 0; j < featDim; ++j) {
                    currDataDst[j] += eFeatsTensor[eid][j] * normFactor;
                }
            }
            lvid++;
        }
    } else { // backward
        unsigned featDim = getFeatDim(c.layer + 1);
        FeatType *featTensor = getVtxFeat(
            savedNNTensors[c.layer + 1]["grad"].getData(), lvid, featDim);
        FeatType *aggTensor = savedNNTensors[c.layer]["aTg"].getData();
        FeatType **eFeatsTensor = savedEdgeTensors[c.layer]["bedge"];

        FeatType *chunkPtr = getVtxFeat(aggTensor, lvid, featDim);
        std::memcpy(chunkPtr, featTensor,
                    sizeof(FeatType) * (limit - lvid) * featDim);
        while (lvid < limit) {
            // Read out data of the current layer of given vertex.
            FeatType *currDataDst = getVtxFeat(aggTensor, lvid, featDim);
            // Apply normalization factor on the current data.
            {
                const EdgeType normFactor = graph.vtxDataVec[lvid];
                for (unsigned i = 0; i < featDim; ++i) {
                    currDataDst[i] *= normFactor;
                }
            }
            // Aggregate from neighbors.
            for (uint64_t eid = graph.backwardAdj.rowPtrs[lvid];
                eid < graph.backwardAdj.rowPtrs[lvid + 1]; ++eid) {
                EdgeType normFactor = graph.backwardAdj.values[eid];
                for (unsigned j = 0; j < featDim; ++j) {
                    currDataDst[j] += eFeatsTensor[eid][j] * normFactor;
                }
            }
            lvid++;
        }
    }
}

void Engine::applyVertexGCN(Chunk &c) {
    resComm->NNCompute(c);
}

void Engine::scatterGCN(Chunk &c) {
    unsigned outputLayer = c.layer;
    unsigned featLayer = c.layer;
    std::string tensorName;
    if (c.dir == PROP_TYPE::FORWARD) {
        outputLayer -= 1;
        tensorName = "h";
    } else {
        outputLayer += 1;
        featLayer += 1;
        tensorName = "grad";
    }
    FeatType *scatterTensor =
        savedNNTensors[outputLayer][tensorName].getData();

    unsigned startId = c.lowBound;
    unsigned endId = c.upBound;
    unsigned featDim = getFeatDim(featLayer);

    std::map<unsigned, std::vector<unsigned>> &ghostMap =
        c.dir == PROP_TYPE::FORWARD ? graph.forwardGhostMap
                                    : graph.backwardGhostMap;

    // Create a series of buckets for batching sendout messages to nodes
    std::vector<unsigned> *batchedIds =
        new std::vector<unsigned>[numNodes];
    for (unsigned lvid = startId; lvid < endId; ++lvid) {
        for (unsigned nid : ghostMap[lvid]) {
            batchedIds[nid].push_back(lvid);
        }
    }

    // batch sendouts similar to the sequential version
    const unsigned BATCH_SIZE = std::max(
        (MAX_MSG_SIZE - DATA_HEADER_SIZE) /
            (sizeof(unsigned) + sizeof(FeatType) * featDim),
        1ul);  // at least send one vertex
    for (unsigned nid = 0; nid < numNodes; ++nid) {
        if (nid == nodeId)
            continue;
        unsigned ghostVCnt = batchedIds[nid].size();
        for (unsigned ib = 0; ib < ghostVCnt; ib += BATCH_SIZE) {
            unsigned sendBatchSize = (ghostVCnt - ib) < BATCH_SIZE
                                   ? (ghostVCnt - ib) : BATCH_SIZE;
            verticesPushOut(nid, sendBatchSize,
                            batchedIds[nid].data() + ib, scatterTensor,
                            featDim, c);
            recvCntLock.lock();
            recvCnt++;
            recvCntLock.unlock();
        }
    }

    delete[] batchedIds;
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
            AVQueue.lock();
            AVQueue.push(c);
            AVQueue.unlock();
        } else {
            abort();
        }

        bs.reset();
    }
    GAQueue.clear();
}

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

        if (gnn_type == GNN::GCN) {
            applyVertexGCN(c);
        } else {
            abort();
        }

        bs.reset();
    }
    AVQueue.clear();
}

void Engine::scatterWorkFunc(unsigned tid) {
    BackoffSleeper bs;
    while (!pipelineHalt) {
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

        Chunk c = SCQueue.top();
        // printLog(nodeId, "SC: Got %s", c.str().c_str());
        unsigned absLayer = getAbsLayer(c, numLayers);
        if (absLayer != layer) {
            layer = absLayer;
            currDir = c.dir;
        }
        SCQueue.pop();
        SCQueue.unlock();

        if (gnn_type == GNN::GCN) {
            scatterGCN(c);
        } else {
            abort();
        }

        SCStashQueue.push_atomic(c);

        bs.reset();
    }
    SCQueue.clear();
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
            applyEdgeGCN(c); // do nothing
            GAQueue.lock();
            GAQueue.push(c);
            GAQueue.unlock();
        }

        bs.reset();
    }
    AEQueue.clear();
}

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
            } else { // Enter next epoch. This is an atomic section
                endTime = getTimer();
                if (currEpoch > 0) { // skip epoch 0
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
                if (currEpoch >= numEpochs) {
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

Engine engine;
