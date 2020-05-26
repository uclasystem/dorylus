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

#ifdef _GPU_ENABLED_
#include "../../common/utils.hpp"
#include "../GPU-Computation/comp_unit.cuh"
#include "../commmanager/GPU_comm.hpp"
static CuMatrix *NormAdjMatrixIn = NULL;
static CuMatrix *NormAdjMatrixOut = NULL;
static CuMatrix *Norms = NULL;
static ComputingUnit cu = ComputingUnit::getInstance();
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

    for (unsigned i = 0; i < numLayers * 2; i++) {
        vecTimeAggregate.push_back(0.0);
        vecTimeApplyVtx.push_back(0.0);
        vecTimeApplyEdg.push_back(0.0);
        vecTimeScatter.push_back(0.0);
        vecTimeLambdaInvoke.push_back(0.0);
        vecTimeLambdaWait.push_back(0.0);
    }

    // save intermediate tensors during forward phase for backward computation.
    vtxNNSavedTensors = new std::vector<Matrix>[numLayers];
    edgNNSavedTensors = new std::vector<Matrix>[numLayers];

    // Save intermediate tensors during forward phase for backward computation.
    savedTensors = new std::vector<Matrix>[numLayers];
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

    preallocate_tensors(GNN::GAT);
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

    delete[] forwardGhostInitData;

    for (int i = 0; i < numLayers; i++) {
        auto &kkv = savedNNTensors[i];
        for (auto &kv : kkv) {
            printLog(nodeId, "free %s", kv.first.c_str());
            if (i == 0 && kv.first == "A") {
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
        FeatType* azTensor = new FeatType[vtxCnt * 1];
        std::memset(azTensor, 0, sizeof(FeatType) * vtxCnt * 1);
        savedNNTensors[layer]["az"] = Matrix(vtxCnt, 1, azTensor);

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
            new FeatType[graph.backwardAdj.nnz * 1];
        std::memset(gradATensor, 0, sizeof(FeatType) * graph.backwardAdj.nnz * 1);
        savedNNTensors[layer]["dA"] =
            Matrix(graph.backwardAdj.nnz, 1, gradATensor);

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
    if (!pipeline) {
        for (unsigned epoch = 0; epoch < numEpochs; ++epoch) {
            double epochStart = getTimer();
            FeatType *predictData = runForward(epoch);
            runBackward(predictData);

            double epochTime = getTimer() - epochStart;
            printLog(nodeId, "Time for epoch %u: %f ms", epoch, epochTime);
            addEpochTime(epochTime);
            if (convergeState == CONVERGE_STATE::DONE) {
                printLog(nodeId, "Early stop at epoch %u", epoch);
                break;
            }
        }
    } else {
        // Run synchronous epoch to setup data
        savedNNTensors[0]["x"] =
            Matrix(graph.localVtxCnt, getFeatDim(0), forwardVerticesInitData);
        savedNNTensors[0]["fghost"] =
            Matrix(graph.srcGhostCnt, getFeatDim(0), forwardGhostInitData);
        savedNNTensors[numLayers - 1]["lab"] = Matrix(
            graph.localVtxCnt, getFeatDim(numLayers), localVerticesLabels);

        // Run one synchronous epoch
        {
            double epochStart = getTimer();
            FeatType *tensor = runForward(0);
            runBackward(tensor);
            double epochTime = getTimer() - epochStart;
            printLog(nodeId, "Time for epoch %u: %f ms", 0, epochTime);
            addEpochTime(epochTime);
        }

        if (nodeId == 0) {
            printLog(nodeId, "Finished SYNCHRONOUS epoch, starting PIPELINE");
        }
        loadChunks();
        // Start pipeline
        runPipeline();

        if (convergeState != CONVERGE_STATE::DONE) {
            for (unsigned epoch = currEpoch; epoch < numEpochs; ++epoch) {
                double epochStart = getTimer();
                FeatType *predictData = runForward(epoch);
                runBackward(predictData);

                double epochTime = getTimer() - epochStart;
                printLog(nodeId, "Time for epoch %u: %f ms", epoch, epochTime);
                addEpochTime(epochTime);
                if (convergeState == CONVERGE_STATE::DONE) {
                    printLog(nodeId, "Early stop at epoch %u", epoch);
                    break;
                }
            }
        }
        printLog(nodeId, "Finished, shutting down...");
    }

    end_time = getCurrentTime();
}

/**
 *
 * Runs a forward propagation phase: (Aggregate -> Lambda Computing -> Ghost
 * Update) -> ( ... ) -> ... Will start a bunch of worker threads and a bunch of
 * data communicator threads.
 *
 */
FeatType *Engine::runForward(unsigned epoch) {
    currEpoch = epoch;
    // Make sure all nodes start running the forward-prop phase.
    if (nodeId == 0) {
        printLog(nodeId, "Epoch %u FORWARD starts...", epoch);
    }

    timeForwardProcess -= getTimer();

    // Create buffer for first-layer aggregation.
    FeatType *inputTensor = forwardVerticesInitData;
    FeatType **eVFeatsTensor = NULL; //savedEdgeTensors[0]["fedge"];
    for (layer = 0; layer < numLayers; ++layer) {
        printLog(nodeId, "Starting layer %u", layer);
        inputTensor =
            applyVertex(inputTensor, graph.localVtxCnt, getFeatDim(layer),
                        getFeatDim(layer + 1), layer == numLayers - 1);
        // For now running two scatters
        //  One for z values, one for az values
        //  Need to fix later. This incurs extra synchornization overhead
        // SCATTER
        printLog(nodeId, "SCATTERING Z");
        inputTensor = savedNNTensors[layer]["z"].getData();
        FeatType* outputTensor = savedNNTensors[layer]["fg_z"].getData();
        eVFeatsTensor =
            scatter(inputTensor, outputTensor, graph.localVtxCnt, getFeatDim(layer + 1));

        printLog(nodeId, "Starting apply edge... /_\\");
        eVFeatsTensor =
            applyEdge(savedNNTensors[layer]["A"].getData(), graph.localInEdgeCnt, 0, eVFeatsTensor,
                      eVFeatsTensor + graph.localInEdgeCnt,
                      getFeatDim(layer + 1), getFeatDim(layer + 1));

        printLog(nodeId, "STARTING AGG");
        eVFeatsTensor = savedEdgeTensors[layer]["fedge"];
        inputTensor = aggregate(eVFeatsTensor, graph.localVtxCnt,
                                getFeatDim(layer + 1), AGGREGATOR::WSUM);
    }

    layer = numLayers - 1;
    printLog(nodeId, "Getting grad mat from layer %u", layer);
    FeatType* predictions = savedNNTensors[layer]["grad"].getData();
    predictions = softmax(inputTensor, predictions, graph.localVtxCnt, getFeatDim(numLayers));

    timeForwardProcess += getTimer();
    // if (nodeId == 0) {
    //     printLog(nodeId, "Epoch %u FORWARD finishes at layer %u.", epoch, layer);
    // }
    // calcAcc(inputTensor, localVerticesLabels, graph.localVtxCnt,
    // getFeatDim(numLayers));

    return predictions;
}

/**
 *
 * Runs a backward propagation phase: (Lambda Computing) -> ( ... ) -> ...
 * Will start a bunch of worker threads and a bunch of data communicator
 * threads.
 *
 */
void Engine::runBackward(FeatType *initGradTensor) {
    // if (nodeId == 0) {
    //     printLog(nodeId, "Epoch %u BACKWARD starts...", currEpoch);
    // }

    timeBackwardProcess -= getTimer();

    // Calculate output loss
    Matrix& labels = savedNNTensors[numLayers - 1]["lab"];
    Matrix& outputDeriv = savedNNTensors[numLayers - 1]["grad"];
    outputDeriv -= labels;

    // Create buffer for first-layer aggregation.
    FeatType *gradTensor = outputDeriv.getData();

    // Pure sequential
    FeatType **eVGradTensor = NULL;
    for (layer = numLayers - 1; layer >= 0; --layer) {
        printLog(nodeId, "SCATTER BACK");
        FeatType* ghostTensor = savedNNTensors[layer]["bg_d"].getData();
        eVGradTensor =
            scatterBackward(gradTensor, ghostTensor, graph.localVtxCnt, getFeatDim(layer + 1));
        printLog(nodeId, "APP EDGE BACK");
        eVGradTensor = applyEdgeBackward(NULL, graph.localOutEdgeCnt, 0,
                                         eVGradTensor + graph.localOutEdgeCnt,
                                         eVGradTensor, getFeatDim(layer + 1),
                                         getFeatDim(layer + 1));
        printLog(nodeId, "AGG BACK");
        gradTensor = aggregateBackward(eVGradTensor, graph.localOutEdgeCnt,
                                       getFeatDim(layer + 1), AGGREGATOR::WSUM);
        gradTensor =
            applyVertexBackward(gradTensor, graph.localVtxCnt,
                                getFeatDim(layer), getFeatDim(layer + 1));
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
void Engine::runPipeline() {
    double stt = getTimer();

    resComm->setAsync(true, currEpoch);

    commHalt = false;
    maxEpoch = 0; // for async/sync switching

    auto ghstRcvr =
        std::bind(&Engine::ghostReceiver, this, std::placeholders::_1);
    std::thread grt(ghstRcvr, 0);

    auto scttrWrkr =
        std::bind(&Engine::scatterWorker, this, std::placeholders::_1);
    std::thread swt(scttrWrkr, 1);

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
    swt.join();
    grt.join();
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

Engine engine;
