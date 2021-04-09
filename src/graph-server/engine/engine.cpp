#include "engine.hpp"
#include "../utils/utils.hpp"

#include <omp.h>

#include <algorithm>
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

    adjustPortsForWorkers();

    // Initialize the node manager and communication manager.
    nodeManager.init(workersFile, myPrIpFile,
                     this);  // NodeManger should go first.

    return;

    nodeId = nodeManager.getMyNodeId();
    numNodes = nodeManager.getNumNodes();
    assert(numNodes <= 256);  // Cluster size limitation.
    outFile += std::to_string(nodeId);
    // Init data ctx with `dThreads` threads for scatter
    commManager.init(nodeManager, mode == LAMBDA ? commThreads : 1);

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
    // HACK FOR NOW: Need to have some reference to 'cu' so I can test
    // if new setup is working. Later will rework to give a chunk to each
    // different GPU
    if (mode == GPU) {  // GPU
        compUnits = std::vector<ComputingUnit>(ngpus);
        for (unsigned gpuId = 0; gpuId < ngpus; ++gpuId) {
            // Giving each process its own GPU for now so each process will
            // have ngpus=1 and assigned its devId through the CLI
            compUnits[gpuId] = ComputingUnit(localId);
        }
        resComm = new GPUComm(this);
    }
    ComputingUnit& cu = compUnits[0];

    numLambdasForward = ngpus;
    printLog(nodeId, "Loading SparseMatrices for GPU");
    NormAdjMatrixIn = new CuMatrix();
    NormAdjMatrixOut = new CuMatrix();
    {
        Matrix onenorms(graph.localVtxCnt, 1, graph.vtxDataVec.data());
        OneNorms = new CuMatrix(onenorms, cu.handle);
        CuMatrix::MemoryPool.erase(OneNorms->devPtr); // don't free
    }
    {
        FeatType *zerobuf = new FeatType[graph.localVtxCnt];
        for (unsigned i = 0; i < graph.localVtxCnt; i++)
            zerobuf[i] = 0;
        Matrix zeronorms(graph.localVtxCnt, 1, zerobuf);
        ZeroNorms = new CuMatrix(zeronorms, cu.handle);
        CuMatrix::MemoryPool.erase(ZeroNorms->devPtr); // don't free
        delete[] zerobuf;
    }
    NormAdjMatrixIn->loadSpCSC(cu.spHandle, graph);
    NormAdjMatrixOut->loadSpCSR(cu.spHandle, graph);

    // TODO: Create chunk based CSC/CSR matrices and One/Zero Norms
//    unsigned vtcsCnt = graph.localVtxCnt;
//    for (unsigned cid = 0; cid < ngpus; ++cid) {
//        unsigned chunkSize =
//            (vtcsCnt + numLambdasForward - 1) / numLambdasForward;
//        unsigned lowBound = cid * chunkSize;
//        unsigned upBound = std::min(lowBound + chunkSize, vtcsCnt);
//
//        unsigned thisChunkSize = upBound - lowBound;
//
//        // Get info to create CSC matrix chunk
//        unsigned long long chunkEdgesSize =
//            graph.forwardAdj.columnPtrs[upBound] - graph.forwardAdj.columnPtrs[lowBound];
//
//        // Write col Ptrs to chunk and shift to be zero based
//        unsigned long long* chunkColPtr = new unsigned long long[thisChunkSize];
//        unsigned long long startColPtr = graph.forwardAdj.columnPtrs[lowBound];
//        for (unsigned u = 0; u < thisChunkSize; ++u) {
//            chunkColPtr[u] = graph.forwardAdj.columnPtrs[u + lowbound] - startColPtr;
//        }
//
//        unsigned rowIdxChunk = new unsigned[chunkEdgeSize];
//        std::map<unsigned, unsigned> vertexChunkIdToNodeLocalId;
//        unsigned chunkVertexId = 0;
//        for (unsigned eid = 0; eid < chunkEdgeSize; ++eid) {
//            unsigned rowId = graph.forwardAdj.rowIdxs[eid];
//            if (vertexChunkIdToNodeLocalId.find(rowId) == vertexChunkIdToNodeLocalId.end()) {
//                vertexChunkIdToNodeLocalId[rowId] = chunkVertexId++;
//            }
//
//            rowIdxChunk[eid] = vertexChunkIdToNodeLocalId[rowId];
//            chunkIdForwardMaps.push_back(vertexChunkIdToNodeLocalId);
//        }
//
//        EdgeType* chunkEdgeValues = new EdgeType[chunkEdgeSize];
//        memcpy(chunkEdgeValues, graph.forwardAdj.values + startColPtr, chunkEdgeSize);
//
//        CSCMatrix chunkCSCMatrix;
//        chunkCSCMatrix.columnCnt = thisChunkSize;
//        chunkCSCMatrix.nnz = chunkEdgeSizes;
//        chunkCSCMatrix.values = chunkEdgeValues;
//        chunkCSCMatrix.columnPtrs = chunkColPtrs;
//        chunkCSCMatrix.rowIdxs = rowIdxChunk;
//        CSCChunks.push_back(chunkCSCMatrix);
//    }
#endif

    // Initialize synchronization utilities.
    recvCnt = 0;
    recvCntLock.init();
    recvCntCond.init(recvCntLock);

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
        if (gnn_type == GNN::GCN) {
            lambdaName = "gcn";
        } else if (gnn_type == GNN::GAT) {
            lambdaName = "gat";
        } else {
            lambdaName = "invalid_lambda_name";
        }

        resComm = new LambdaComm(this);
    }
#endif
#ifdef _CPU_ENABLED_
    if (mode == CPU) {  // CPU
        resComm = new CPUComm(this);
    }
#endif
//#ifdef _GPU_ENABLED_
//
//#endif

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

    recvCntLock.destroy();
    recvCntCond.destroy();

    if (nodeId == 0) {
        weightComm->shutdown();
        delete weightComm;
    }
    delete resComm;

    // delete[] forwardVerticesInitData;
    // delete[] forwardGhostInitData;
    // delete[] localVerticesLabels;
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
    pipelineHalt = false;
    unsigned commThdCnt = commThreads;
    // unsigned commThdCnt = std::max(2u, cThreads / 4);

    auto gaWrkrFunc =
        std::bind(&Engine::gatherWorkFunc, this, std::placeholders::_1);
    ThreadVector gaWrkrThds;
    for (unsigned tid = 0; tid < aggThreads; ++tid) {
        gaWrkrThds.push_back(std::thread(gaWrkrFunc, 2 + tid));
    }
    auto avWrkrFunc =
        std::bind(&Engine::applyVertexWorkFunc, this, std::placeholders::_1);
    ThreadVector avWrkrThds;
    for (unsigned tid = 0; tid < applyThreads; ++tid) {
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
    for (unsigned tid = 0; tid < applyThreads; ++tid) {
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
    for (unsigned tid = 0; tid < aggThreads; ++tid) {
        gaWrkrThds[tid].join();
    }
    for (unsigned tid = 0; tid < applyThreads; ++tid) {
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

Engine engine;
