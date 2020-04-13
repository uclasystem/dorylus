#include <algorithm>
#include <iostream>
#include <fstream>
#include <cmath>
#include <boost/program_options.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/algorithm/string/classification.hpp>    // Include boost::for is_any_of.
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/trim.hpp>
#include <string>
#include <thread>
#include <cstdlib>
#include <omp.h>
#include <cerrno>

#include <iomanip>
#include <sstream>

#include "engine.hpp"
#include "dataloader.hpp"
#include <unordered_set>

#ifdef _GPU_ENABLED_
#include "../commmanager/GPU_comm.hpp"
#include "../GPU-Computation/comp_unit.cuh"
#include "../../common/utils.hpp"
static CuMatrix *NormAdjMatrixIn = NULL;
static CuMatrix *NormAdjMatrixOut = NULL;
static ComputingUnit cu = ComputingUnit::getInstance();
#else
#include "../commmanager/lambda_comm.hpp"
#endif

// ======== Debug utils ========
typedef std::vector< std::vector<unsigned> > opTimes;

// Outputs a string to a file
void outputToFile(std::ofstream& outfile, std::string str) {
    fileMutex.lock();
    outfile.write(str.c_str(), str.size());
    outfile << std::endl << std::flush;
    fileMutex.unlock();
}
// ======== END Debug File utils ========


/**
 *
 * Initialize the engine with the given command line arguments.
 *
 */
void
Engine::init(int argc, char *argv[]) {
    printLog(404, "Engine starts initialization...");
    timeInit = -getTimer();

    parseArgs(argc, argv);

    // Initialize the node manager and communication manager.
    nodeManager.init(dshMachinesFile, myPrIpFile, myPubIpFile);    // NodeManger should go first.

    nodeId = nodeManager.getMyNodeId();
    numNodes = nodeManager.getNumNodes();
    assert(numNodes <= 256);    // Cluster size limitation.
    outFile += std::to_string(nodeId);
    commManager.init(nodeManager);

    // Set number of layers and number of features in each layer. Also store the prefix sum of config for offset querying use.
    readLayerConfigFile(layerConfigFile);
    numLayers = layerConfig.size() - 1;

    std::string graphFile = datasetDir + "graph." + std::to_string(nodeId) + ".bin";
    // detect whether preprocessed
    {
        bool forcePreprocess = false;
        std::ifstream gfile(graphFile.c_str(), std::ios::binary);
        if (!gfile.good() || forcePreprocess) {
            DataLoader dl(datasetDir, nodeId, numNodes, undirected);
            dl.preprocess();
        }
    }
    graph.init(graphFile);
#ifdef _GPU_ENABLED_
    printLog(nodeId, "Loading SparseMatrices for GPU");
    NormAdjMatrixIn = new CuMatrix();
    NormAdjMatrixOut = new CuMatrix();
    NormAdjMatrixIn->loadSpCSC(cu.spHandle, graph);
    NormAdjMatrixOut->loadSpCSR(cu.spHandle, graph);
#endif

    printGraphMetrics();

    // save intermediate tensors during forward phase for backward computation.
    vtxNNSavedTensors = new std::vector<Matrix>[numLayers];
    edgNNSavedTensors = new std::vector<Matrix>[numLayers];

    // Save intermediate tensors during forward phase for backward computation.
    savedTensors = new std::vector<Matrix>[numLayers];
    savedNNTensors.resize(numLayers);
    savedEdgeTensors.resize(numLayers);

    // Track the number of chunks finished at each epoch;
    if (staleness != UINT_MAX) numFinishedEpoch.resize(staleness+1);

    // Init it here for collecting data when reading files
    forwardVerticesInitData = new FeatType[getFeatDim(0) * graph.localVtxCnt];
    forwardGhostInitData = new FeatType[getFeatDim(0) * graph.srcGhostCnt];
    // Create labels storage area. Read in labels and store as one-hot format.
    localVerticesLabels = new FeatType[layerConfig[numLayers] * graph.localVtxCnt];

    // Read in initial feature values (input features) & labels.
    readFeaturesFile(featuresFile);
    readLabelsFile(labelsFile);

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
        weightComm->updateChunkCnt(numNodes * numLambdasForward); // now set up weight servers only once
    } else {
        weightComm = NULL;
    }
    if (mode == LAMBDA) { // Lambda
        resComm = new LambdaComm(this);
    } else if (mode == GPU) { // GPU
    } else if (mode == CPU) { // CPU
    }

    timeForwardProcess = 0.0;
    timeInit += getTimer();
    printLog(nodeId, "Engine initialization complete.");

    preallocate_tensors(GNN::GCN);
    start_time = getCurrentTime();
}


void
Engine::preallocate_tensors(GNN gnn_type) {
    switch (gnn_type) {
        case GNN::GCN:
            preallocateGCN();
            break;
        default:
            printLog(nodeId, "Unrecognized benchmark type");
    }
}


void
Engine::preallocateGCN() {
    unsigned vtxCnt = graph.localVtxCnt;

    // Store input tesnors
    savedNNTensors[0]["x"] = Matrix(vtxCnt, getFeatDim(0), forwardVerticesInitData);
    savedNNTensors[0]["fg"] = Matrix(graph.srcGhostCnt, getFeatDim(0),
      forwardGhostInitData);
    savedNNTensors[numLayers-1]["lab"] = Matrix(vtxCnt, getFeatDim(numLayers),
      localVerticesLabels);

    FeatType** eVFeatsTensor = srcVFeats2eFeats(forwardVerticesInitData,
      forwardGhostInitData, vtxCnt, getFeatDim(0));
    savedEdgeTensors[0]["fedge"] = eVFeatsTensor;

    // forward tensor allocation
    for (layer = 0; layer < numLayers; ++layer) {
        unsigned featDim = getFeatDim(layer);
        unsigned nextFeatDim = getFeatDim(layer+1);

        // GATHER TENSORS
        FeatType* ahTensor = new FeatType[vtxCnt * featDim];
        savedNNTensors[layer]["ah"] = Matrix("ah", vtxCnt, featDim, ahTensor);

        // APPLY TENSORS
        if (layer < numLayers - 1) {
            FeatType* zTensor = new FeatType[vtxCnt * nextFeatDim];
            FeatType* hTensor = new FeatType[vtxCnt * nextFeatDim];

            savedNNTensors[layer]["z"] = Matrix(vtxCnt, nextFeatDim, zTensor);
            savedNNTensors[layer]["h"] = Matrix(vtxCnt, nextFeatDim, hTensor);

            // SCATTER TENSORS
            FeatType* ghostTensor = new FeatType[graph.srcGhostCnt * nextFeatDim];
            savedNNTensors[layer+1]["fg"] = Matrix(graph.srcGhostCnt,
              nextFeatDim, ghostTensor);

            FeatType** edgeTensor = srcVFeats2eFeats(hTensor, ghostTensor,
              vtxCnt, nextFeatDim);
            savedEdgeTensors[layer+1]["fedge"] = edgeTensor;
        }
    }

    // backward tensor allocation
    for (layer = numLayers - 1; layer > 0; --layer) {
        unsigned featDim = getFeatDim(layer);

        // APPLY TENSORS
        FeatType* gradTensor = new FeatType[vtxCnt * featDim];
        savedNNTensors[layer]["grad"] = Matrix("grad", vtxCnt, featDim, gradTensor);

        // SCATTER TENSORS
        FeatType* ghostTensor = new FeatType[graph.dstGhostCnt * featDim];
        savedNNTensors[layer-1]["bg"] = Matrix(graph.dstGhostCnt, featDim,
          ghostTensor);

        FeatType** eFeats = dstVFeats2eFeats(gradTensor, ghostTensor, vtxCnt,
          featDim);
        savedEdgeTensors[layer-1]["bedge"] = eFeats;

        // GATHER TENSORS
        FeatType* aTgTensor = new FeatType[vtxCnt * featDim];
        savedNNTensors[layer-1]["aTg"] = Matrix(vtxCnt, featDim, aTgTensor);
    }
}


/**
 *
 * Destroy the engine.
 *
 */
void
Engine::destroy() {
    printLog(nodeId, "Destroying the engine...");

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
    if (mode == LAMBDA) {
        delete resComm;
    }

    delete[] forwardVerticesInitData;
    delete[] forwardGhostInitData;

    delete[] localVerticesLabels;
}


/**
 *
 * Whether I am the master mode or not.
 *
 */
bool
Engine::master() {
    return nodeManager.amIMaster();
}

void
Engine::makeBarrier() {
    nodeManager.barrier();
}

/**
 *
 * How many epochs to run for
 *
 */
unsigned
Engine::getNumEpochs() {
    return numEpochs;
}

/**
 *
 * Add a new epoch time to the list of epoch times
 *
 */
void
Engine::addEpochTime(double epochTime) {
    epochTimes.push_back(epochTime);
}

/**
 *
 * How many epochs to run before validation
 *
 */
unsigned
Engine::getValFreq() {
    return valFreq;
}

/**
 *
 * Return the ID of this node
 *
 */
unsigned
Engine::getNodeId() {
    return nodeId;
}

/**
 *
 * Runs a forward propagation phase: (Aggregate -> Lambda Computing -> Ghost Update) -> ( ... ) -> ...
 * Will start a bunch of worker threads and a bunch of data communicator threads.
 *
 */
FeatType*
Engine::runForward(unsigned epoch) {
    currEpoch = epoch;
    // Make sure all nodes start running the forward-prop phase.
    printLog(nodeId, "Engine starts running FORWARD...");

    timeForwardProcess -= getTimer();

    // Create buffer for first-layer aggregation.
    FeatType *inputTensor = forwardVerticesInitData;
    forwardGhostVerticesDataIn = forwardGhostInitData;
    // For sequential invocation of operations use this block
    //FeatType **eVFeatsTensor = srcVFeats2eFeats(inputTensor, forwardGhostInitData, graph.localVtxCnt, getFeatDim(layer));
    FeatType** eVFeatsTensor = savedEdgeTensors[0]["fedge"];
    for (layer = 0; layer < numLayers; ++layer) {
        inputTensor = aggregate(eVFeatsTensor, graph.localVtxCnt, getFeatDim(layer), AGGREGATOR::WSUM);
        inputTensor = applyVertex(inputTensor, graph.localVtxCnt,
          getFeatDim(layer), getFeatDim(layer + 1), layer == numLayers-1);
        if (layer < numLayers - 1) { // don't need scatter at the last layer.
            eVFeatsTensor = scatter(inputTensor, graph.localVtxCnt, getFeatDim(layer + 1));
            eVFeatsTensor = applyEdge(NULL, graph.localInEdgeCnt, 0,
                                    eVFeatsTensor, eVFeatsTensor + graph.localInEdgeCnt,
                                    getFeatDim(layer + 1), getFeatDim(layer + 1));
        }
    }

    timeForwardProcess += getTimer();
    printLog(nodeId, "Engine completes FORWARD at layer %u.", layer);
    //calcAcc(inputTensor, localVerticesLabels, graph.localVtxCnt, getFeatDim(numLayers));

    return inputTensor;
}


/**
 *
 * Runs a backward propagation phase: (Lambda Computing) -> ( ... ) -> ...
 * Will start a bunch of worker threads and a bunch of data communicator threads.
 *
 */
void
Engine::runBackward(FeatType *initGradTensor) {
    printLog(nodeId, "Engine starts running BACKWARD...");

    timeBackwardProcess -= getTimer();

    // Create buffer for first-layer aggregation.
    FeatType *gradTensor = initGradTensor;

    // Pure sequential
    FeatType **eVGradTensor = NULL;
    for (layer = numLayers - 1; layer > 0; --layer) {
        eVGradTensor = scatterBackward(gradTensor, graph.localVtxCnt, getFeatDim(layer));
        eVGradTensor = applyEdgeBackward(NULL, graph.localOutEdgeCnt, 0,
                                    eVGradTensor + graph.localOutEdgeCnt, eVGradTensor,
                                    getFeatDim(layer), getFeatDim(layer));
        gradTensor = aggregateBackward(eVGradTensor, graph.localOutEdgeCnt, getFeatDim(layer), AGGREGATOR::WSUM);
        gradTensor = applyVertexBackward(gradTensor, graph.localVtxCnt, getFeatDim(layer - 1), getFeatDim(layer));
    }

    timeBackwardProcess += getTimer();
    printLog(nodeId, "Engine completes BACKWARD at %u.", layer);
}


void
Engine::runGCN() {
    // Run synchronous epoch to setup data
    savedNNTensors[0]["x"] = Matrix(graph.localVtxCnt, getFeatDim(0), forwardVerticesInitData);
    savedNNTensors[0]["fghost"] = Matrix(graph.srcGhostCnt, getFeatDim(0), forwardGhostInitData);
    savedNNTensors[numLayers-1]["lab"] = Matrix(graph.localVtxCnt, getFeatDim(numLayers), localVerticesLabels);

    // Run one synchronous epoch
    FeatType* tensor = runForward(0);
    runBackward(tensor);

    printLog(nodeId, "Finished SYNCHRONOUS epoch, starting PIPELINE");
    loadChunks();
    resComm->setAsync(true);
    // Start pipeline
    runPipeline();
}


/**
 * Run the deep-pipeline version where all stages happen in parallel
 */
void
Engine::runPipeline() {
    commHalt = false;
    auto ghstRcvr = std::bind(&Engine::ghostReceiver, this, std::placeholders::_1);
    std::thread t(ghstRcvr, 0);
    t.detach();

    auto scttrWrkr = std::bind(&Engine::scatterWorker, this, std::placeholders::_1);
    std::thread t2(scttrWrkr, 1);
    t2.detach();

    aggregator(2);
}


/**
 *
 * Write output stuff to the tmp directory for every local vertex.
 * Write engine timing metrics to the logfile.
 *
 */
void
Engine::output() {
    std::ofstream outStream(outFile.c_str());
    if (!outStream.good())
        printLog(nodeId, "Cannot open output file: %s [Reason: %s]", outFile.c_str(), std::strerror(errno));

    assert(outStream.good());

    //
    // The following are labels outputing.
    //
    // for (Vertex& v : graph.getVertices()) {
    //     outStream << v.getGlobalId() << ": ";
    //     FeatType *labelsPtr = localVertexLabelsPtr(v.getLocalId());
    //     for (unsigned i = 0; i < layerConfig[numLayers]; ++i)
    //         outStream << labelsPtr[i] << " ";
    //     outStream << std::endl;
    // }

    //
    // The following are timing results outputing.
    //
    // outStream << "I: " << timeInit << std::endl;
    // for (unsigned i = 0; i < numLayers; ++i) {
    //     outStream << "A: " << vecTimeAggregate[i] << std::endl
    //               << "L: " << vecTimeApplyVtx[i] << std::endl
    //               << "G: " << vecTimeScatter[i] << std::endl;
    // }
    // outStream << "B: " << timeBackwardProcess << std::endl;

    std::time_t end_time = getCurrentTime();

    char outBuf[1024];
    sprintf(outBuf, "<EM>: Run start time: ");
    outStream << outBuf << std::ctime(&start_time);
    sprintf(outBuf, "<EM>: Run end time: ");
    outStream << outBuf << std::ctime(&end_time);

    sprintf(outBuf, "<EM>: Using %u forward lambdas and %u backward lambdas",
            numLambdasForward, numLambdasBackward);
    outStream << outBuf << std::endl;

    sprintf(outBuf, "<EM>: Initialization takes %.3lf ms", timeInit);
    outStream << outBuf << std::endl;
    if (!pipeline) {
        sprintf(outBuf, "<EM>: Forward:  Time per stage:");
        outStream << outBuf << std::endl;
        for (unsigned i = 0; i < numLayers; ++i) {
            sprintf(outBuf, "<EM>    Aggregation   %2u  %.3lf ms", i, vecTimeAggregate[i] / (float)numEpochs);
            outStream << outBuf << std::endl;
            sprintf(outBuf, "<EM>    ApplyVertex   %2u  %.3lf ms", i, vecTimeApplyVtx[i] / (float)numEpochs);
            outStream << outBuf << std::endl;
            sprintf(outBuf, "<EM>    Scatter       %2u  %.3lf ms", i, vecTimeScatter[i] / (float)numEpochs);
            outStream << outBuf << std::endl;
            sprintf(outBuf, "<EM>    ApplyEdge     %2u  %.3lf ms", i, vecTimeApplyEdg[i] / (float)numEpochs);
            outStream << outBuf << std::endl;
        }
    }
    sprintf(outBuf, "<EM>: Total forward-prop time %.3lf ms", timeForwardProcess / (float)numEpochs);
    outStream << outBuf << std::endl;

    if (!pipeline) {
        sprintf(outBuf, "<EM>: Backward: Time per stage:");
        outStream << outBuf << std::endl;
        for (unsigned i = numLayers; i < 2 * numLayers; i++) {
            sprintf(outBuf, "<EM>    Aggregation   %2u  %.3lf ms", i, vecTimeAggregate[i] / (float)numEpochs);
            outStream << outBuf << std::endl;
            sprintf(outBuf, "<EM>    ApplyVertex   %2u  %.3lf ms", i, vecTimeApplyVtx[i] / (float)numEpochs);
            outStream << outBuf << std::endl;
            sprintf(outBuf, "<EM>    Scatter       %2u  %.3lf ms", i, vecTimeScatter[i] / (float)numEpochs);
            outStream << outBuf << std::endl;
            sprintf(outBuf, "<EM>    ApplyEdge     %2u  %.3lf ms", i, vecTimeApplyEdg[i] / (float)numEpochs);
            outStream << outBuf << std::endl;
        }
    }
    sprintf(outBuf, "<EM>: Total backward-prop time %.3lf ms", timeBackwardProcess);
    outStream << outBuf << std::endl;

    double sum = 0.0;
    for (double& d : epochTimes) sum += d;
    sprintf(outBuf, "<EM>: Average epoch time %.3lf ms", sum / (float)epochTimes.size());
    outStream << outBuf << std::endl;
    sprintf(outBuf, "<EM>: Final accuracy %.3lf", accuracy);
    outStream << outBuf << std::endl;
    sprintf(outBuf, "Relaunched Lambda Cnt: %u", resComm->getRelaunchCnt());
    outStream << outBuf << std::endl;

    // Write benchmarking results to log file.
    if (master()) {
        assert(vecTimeAggregate.size() == 2 * numLayers);
        assert(vecTimeApplyVtx.size() == 2 * numLayers);
        assert(pipeline || vecTimeScatter.size() == 2 * numLayers);
        assert(pipeline || vecTimeApplyEdg.size() == 2 * numLayers);
        printEngineMetrics();
    }
}


#ifdef _GPU_ENABLED_
FeatType *Engine::aggregate(FeatType *vtcsTensor, unsigned vtcsCnt, unsigned featDim) {
    auto t0 = gtimers.getTimer("Memcpy2GPUForwardTimer");
    auto t1 = gtimers.getTimer("AggForwardTimer");
    auto t2 = gtimers.getTimer("ComputeTransForwardTimer");
    auto t3 = gtimers.getTimer("Memcpy2RAMForwardTimer");

    double sttTimer = getTimer();
    FeatType *outputTensor = new FeatType [(vtcsCnt) * featDim];
    CuMatrix feat;
    t0->start();
    feat.loadSpDense(vtcsTensor, forwardGhostVerticesDataIn,
                     graph.localVtxCnt, graph.srcGhostCnt,
                     featDim);
    cudaDeviceSynchronize();
    t0->stop();
    t1->start();
    CuMatrix out = cu.aggregate(*NormAdjMatrixIn, feat);
    cudaDeviceSynchronize();
    t1->stop();
    t2->start();
    out = out.transpose();
    cudaDeviceSynchronize();
    t2->stop();
    t3->start();
    out.setData(outputTensor);
    out.updateMatrixFromGPU();
    t3->stop();

    currId = vtcsCnt;

    if (layer > 0) {
        delete[] forwardGhostVerticesDataIn;
        delete[] vtcsTensor;
    }

    if (vecTimeAggregate.size() < numLayers) {
        vecTimeAggregate.push_back(getTimer() - sttTimer);
    } else {
        vecTimeAggregate[layer] += getTimer() - sttTimer;
    }

    return outputTensor;
}
#else
FeatType* Engine::aggregate(FeatType **edgsTensor, unsigned edgsCnt, unsigned featDim, AGGREGATOR aggregator) {
    double sttTimer = getTimer();
    // AH
    FeatType* outputTensor = savedNNTensors[layer]["ah"].getData();
    FeatType* hTensor = NULL;
    if (layer == 0) {
        hTensor = savedNNTensors[layer]["x"].getData();
    } else {
        hTensor = savedNNTensors[layer-1]["h"].getData();
    }
    currId = 0;

    switch (aggregator) {
        case (AGGREGATOR::WSUM): {
            memcpy(outputTensor, hTensor, sizeof(FeatType) * graph.localVtxCnt * featDim);

            AggOPArgs args = {outputTensor, edgsTensor, graph.localVtxCnt, edgsCnt, featDim};
            auto computeFn = std::bind(&Engine::aggregateCompute, this, std::placeholders::_1, std::placeholders::_2);

            computePool->perform(computeFn, &args);
            computePool->sync();
            break;
        }
        default:
            printLog(nodeId, "Invalid Aggregator %d.", aggregator);
            break;
    }

    if (vecTimeAggregate.size() < numLayers) {
        vecTimeAggregate.push_back(getTimer() - sttTimer);
    } else {
        vecTimeAggregate[layer] += getTimer() - sttTimer;
    }
    return outputTensor;
}
#endif // _GPU_ENABLED_

FeatType *
Engine::applyVertex(FeatType *vtcsTensor, unsigned vtcsCnt, unsigned inFeatDim,
  unsigned outFeatDim, bool lastLayer) {
    double sttTimer = getTimer();
    assert(vtcsCnt == graph.localVtxCnt);

    FeatType* outputTensor = NULL;
    if (lastLayer) {
        outputTensor = savedNNTensors[layer]["grad"].getData();
    } else {
        outputTensor = savedNNTensors[layer]["h"].getData();
    }

    // Start a new lambda communication context.
    if (mode == LAMBDA) {
        double invTimer = getTimer();
        const unsigned chunkSize = (vtcsCnt + numLambdasForward - 1) / numLambdasForward;
        unsigned availLambdaId = 0;
        while (availLambdaId < numLambdasForward) {
            unsigned lowBound = availLambdaId * chunkSize;
            unsigned upBound = std::min(lowBound + chunkSize, vtcsCnt);
            Chunk chunk {availLambdaId, lowBound, upBound, layer, PROP_TYPE::FORWARD, currEpoch, true}; // epoch is not useful in sync version
            resComm->NNCompute(chunk);

            availLambdaId++;
        }
        if (vecTimeLambdaInvoke.size() < numLayers) {
            vecTimeLambdaInvoke.push_back(getTimer() - invTimer);
        } else {
            vecTimeLambdaInvoke[layer] += getTimer() - invTimer;
        }
        double waitTimer = getTimer();
        resComm->NNSync();
        if (vecTimeLambdaWait.size() < numLayers) {
            vecTimeLambdaWait.push_back(getTimer() - waitTimer);
        } else {
            vecTimeLambdaWait[layer] += getTimer() - waitTimer;
        }
    }
    // if in GPU mode we launch gpu computation here and wait the results
    else {} // TODO: (YIFAN) support for GPU/CPU

    if (vecTimeApplyVtx.size() < numLayers) {
        vecTimeApplyVtx.push_back(getTimer() - sttTimer);
    } else {
        vecTimeApplyVtx[layer] += getTimer() - sttTimer;
    }

    return outputTensor;
}


FeatType **
Engine::applyEdge(EdgeType *edgsTensor, unsigned edgsCnt, unsigned eFeatDim, FeatType **eSrcVtcsTensor, FeatType **eDstVtcsTensor, unsigned inFeatDim, unsigned outFeatDim) {
    double sttTimer = getTimer();

    // do nothing
    FeatType **outputTensor = eSrcVtcsTensor;
    eSrcVtcsTensor = NULL;

    if (vecTimeApplyEdg.size() < numLayers) {
        vecTimeApplyEdg.push_back(getTimer() - sttTimer);
    } else {
        vecTimeApplyEdg[layer] += getTimer() - sttTimer;
    }

    return outputTensor;
}

FeatType **
Engine::scatter(FeatType *vtcsTensor, unsigned vtcsCnt, unsigned featDim) {
    double sttTimer = getTimer();

    // Start data communicators.
    commHalt = false;
    recvCnt = 0;
    forwardGhostVerticesDataOut = savedNNTensors[layer+1]["fg"].getData();
    if (forwardGhostVerticesDataOut == NULL) {
        printLog(nodeId, "Forward scatter buffer pointer is NULL");
    }
    auto fgr_fp = std::bind(&Engine::forwardGhostReceiver, this,
      std::placeholders::_1);
    dataPool->perform(fgr_fp);

    sendForwardGhostUpdates(vtcsTensor, featDim);

    // TODO: (YIFAN) we can optimize this to extend comm protocol. Mark the last packet sent so this node knows when to exit ghostCommunicator.
    nodeManager.barrier();

    commHalt = true;
    // Join all data communicators.
    dataPool->sync();

    //FeatType **edgsTensor = srcVFeats2eFeats(vtcsTensor, forwardGhostVerticesDataIn, vtcsCnt, featDim);
    FeatType** edgsTensor = savedEdgeTensors[layer+1]["fedge"];
    vtcsTensor = NULL;

    if (vecTimeScatter.size() < numLayers) {
        vecTimeScatter.push_back(getTimer() - sttTimer);
    } else {
        vecTimeScatter[layer] += getTimer() - sttTimer;
    }

    return edgsTensor;
}

FeatType*
Engine::fusedGAS(FeatType* vtcsTensor, unsigned vtcsCnt, unsigned inFeatDim,
  unsigned outFeatDim, bool scatter) {
    return NULL;
    // double sttTimer = getTimer();
    // // Check just to make sure partition ranges are empty
    // // before starting
    // consumerQueueLock.lock();
    // while (!rangesToScatter.empty()) rangesToScatter.pop();
    // consumerQueueLock.unlock();

    // // Start data receivers
    // commHalt = false;
    // // Forward declaration to ensure pointer remains in scope
    // FeatType *outputTensor = new FeatType[vtcsCnt * outFeatDim];
    // FeatType *zTensor = new FeatType[vtcsCnt * outFeatDim];
    // auto fgr_fp = std::bind(&Engine::forwardGhostReceiver, this, std::placeholders::_1);
    // auto fgu_fp = std::bind(&Engine::pipelineForwardGhostUpdates, this,
    //                 std::placeholders::_1, std::placeholders::_2);
    // std::thread scatterThread;
    // if (scatter) {
    //     forwardGhostVerticesDataOut = new FeatType[graph.srcGhostCnt * outFeatDim];
    //     dataPool->perform(fgr_fp);
    //     scatterThread = std::thread(fgu_fp, outputTensor, outFeatDim);
    // }

    // // Prepare for gather phase
    // FeatType *gatheredTensor = new FeatType[vtcsCnt * inFeatDim];
    // currId = 0;
    // AggOPArgs args = {gatheredTensor, vtcsTensor, vtcsCnt, inFeatDim};
    // auto computeFn = std::bind(&Engine::aggregateCompute, this, std::placeholders::_1, std::placeholders::_2);

    // // Start gathering
    // computePool->perform(computeFn, &args);

    // // Prepare for applyVertex phase
    // bool saveInput = true;
    // if (saveInput) {
    //     vtxNNSavedTensors[layer].push_back(Matrix(vtcsCnt, inFeatDim, gatheredTensor));
    // }
    // vtxNNSavedTensors[layer].push_back(Matrix(vtcsCnt, outFeatDim, zTensor));
    // Matrix inputTensor_ = Matrix(vtcsCnt, inFeatDim, gatheredTensor);
    // Matrix outputTensor_ = Matrix(vtcsCnt, outFeatDim, outputTensor);
    // resComm->newContext(layer, inputTensor_, outputTensor_, vtxNNSavedTensors, scatter);

    // // Start applyVertex phase
    // unsigned currLambdaId = 0;
    // if (mode == LAMBDA) {
    //     const unsigned lambdaChunkSize = (vtcsCnt + numLambdasForward - 1) / numLambdasForward;
    //     unsigned availChunkSize = lambdaChunkSize;
    //     while (currId < vtcsCnt) {
    //         unsigned lvid = currId;
    //         while (lvid > availChunkSize) {
    //             resComm->applyVertexForward(layer, currLambdaId, layer == numLayers - 1);
    //             ++currLambdaId;
    //             availChunkSize += lambdaChunkSize;
    //         }
    //         usleep(5000); // wait 5ms and then check again
    //     }
    // }
    // computePool->sync();
    // if (mode != LAMBDA) {
    //     resComm->requestForward(layer, layer == numLayers - 1);
    // } else {
    //     while (currLambdaId < numLambdasForward) {
    //         resComm->applyVertexForward(layer, currLambdaId, layer == numLayers - 1);
    //         ++currLambdaId;
    //     }
    //     resComm->waitResForward(layer, layer == numLayers - 1);
    // }

    // // Wait for all remote schedulings sent by me to be handled.
    // if (scatter) {
    //     scatterThread.join();
    // }
    // nodeManager.barrier();
    // commHalt = true;
    // dataPool->sync();

    // // Post-processing for applyVertex phase & clean up
    // bool saveOutput = true;
    // if (saveOutput) {
    //     FeatType *outTensorCpy = new FeatType[vtcsCnt * outFeatDim];
    //     memcpy(outTensorCpy, outputTensor, vtcsCnt * outFeatDim * sizeof(FeatType));
    //     vtxNNSavedTensors[layer].push_back(Matrix(vtcsCnt, outFeatDim, outTensorCpy));
    // }
    // if (saveInput) {
    //     gatheredTensor = NULL;
    // } else {
    //     delete[] gatheredTensor;
    // }

    // // Clean up the gather phase
    // if (layer > 0) {
    //     delete[] forwardGhostVerticesDataIn;
    //     delete[] vtcsTensor;
    // }

    // if (vecTimeAggregate.size() < numLayers) {
    //     vecTimeAggregate.push_back(getTimer() - sttTimer);
    // } else {
    //     vecTimeAggregate[layer] += getTimer() - sttTimer;
    // }

    // // Set the scattered output as the input for next aggregation phase
    // forwardGhostVerticesDataIn = forwardGhostVerticesDataOut;

    // return outputTensor;
}


/////////////////////////////////////////////////////////
// Below are private forward functions for the engine. //
/////////////////////////////////////////////////////////
void Engine::aggregateCompute(unsigned tid, void *args) {
    FeatType *outputTensor = ((AggOPArgs *) args)->outputTensor;
    FeatType **eVFeatsTensor = ((AggOPArgs *) args)->inputTensor;
    const unsigned vtcsCnt = ((AggOPArgs *) args)->vtcsCnt;
    // const unsigned edgsCnt = ((AggOPArgs *) args)->edgsCnt;
    const unsigned featDim = ((AggOPArgs *) args)->featDim;

    unsigned lvid = 0;
    while (currId < vtcsCnt) {
        lvid = __sync_fetch_and_add(&currId, 1);
        if (lvid < vtcsCnt) {
            forwardAggregateFromNeighbors(lvid, outputTensor, eVFeatsTensor, featDim);
        }
    }
}

void
Engine::aggregateChunk(Chunk& c) {
    unsigned lvid = c.lowBound;
    unsigned limit = c.upBound;
    unsigned featDim = getFeatDim(c.layer);

    FeatType* featTensor = NULL;
    if (c.layer == 0) featTensor = getVtxFeat(savedNNTensors[c.layer]["x"].getData(), lvid, featDim);
    else featTensor = getVtxFeat(savedNNTensors[c.layer-1]["h"].getData(), lvid, featDim);

    FeatType* aggTensor = savedNNTensors[c.layer]["ah"].getData();
    FeatType** eFeatsTensor = savedEdgeTensors[c.layer]["fedge"];

    FeatType* chunkPtr = getVtxFeat(aggTensor, lvid, featDim);
    std::memcpy(chunkPtr, featTensor, sizeof(FeatType) * (limit - lvid) * featDim);
    while (lvid < limit) {
        forwardAggregateFromNeighbors(lvid++, aggTensor, eFeatsTensor, getFeatDim(c.layer));
    }
}


/**
 *
 * Aggregate featDim feature values starting from offset from all neighbors (including self). Then write the results to the
 * data buffer area for serialization. The results are to be used for being sent to lambda threads.
 *
 */
inline void
Engine::forwardAggregateFromNeighbors(unsigned lvid, FeatType *outputTensor, FeatType **inputTensor, unsigned featDim) {
    // Read out data of the current layer of given vertex.
    FeatType *currDataDst = getVtxFeat(outputTensor, lvid, featDim);

    // Apply normalization factor on the current data.
    {
        const EdgeType normFactor = graph.vtxDataVec[lvid];
        for (unsigned i = 0; i < featDim; ++i) {
            currDataDst[i] *= normFactor;
        }
    }

    // Aggregate from incoming neighbors.
    for (unsigned long long eid = graph.forwardAdj.columnPtrs[lvid]; eid < graph.forwardAdj.columnPtrs[lvid + 1]; ++eid) {
        EdgeType normFactor = graph.forwardAdj.values[eid];
        for (unsigned j = 0; j < featDim; ++j) {
            currDataDst[j] += inputTensor[eid][j] * normFactor;
        }
    }
}

// Loop through all local vertices and do the data send out work.
// If there are any remote edges for a vertex, should send this vid to
// other nodes for their ghost's update.
inline void
Engine::sendForwardGhostUpdates(FeatType *inputTensor, unsigned featDim) {
    bool batchFlag = true;
    unsigned BATCH_SIZE = std::max(((batchFlag ? MAX_MSG_SIZE : 4096) - DATA_HEADER_SIZE) /
                                   (sizeof(unsigned) + sizeof(FeatType) * featDim), 1ul); // at least send one vertex
    for (unsigned nid = 0; nid < numNodes; ++nid) {
        if (nid == nodeId) {
            continue;
        }

        unsigned forwardGhostVCnt = graph.forwardLocalVtxDsts[nid].size();
        for (unsigned ib = 0; ib < forwardGhostVCnt; ib += BATCH_SIZE) {
            unsigned sendBatchSize = (forwardGhostVCnt - ib) < BATCH_SIZE ? (forwardGhostVCnt - ib) : BATCH_SIZE;

            forwardVerticesPushOut(nid, sendBatchSize, graph.forwardLocalVtxDsts[nid].data() + ib, inputTensor, featDim);
            recvCntLock.lock();
            recvCnt++;
            recvCntLock.unlock();
        }
    }
    // Wait for all remote schedulings sent by me to be handled.
    recvCntLock.lock();
    if (recvCnt > 0) {
        recvCntCond.wait();
    }
    recvCntLock.unlock();
}

// inputTensor = activation output tensor
inline void
Engine::pipelineForwardGhostUpdates(unsigned tid) {
//    int failedTrials = 0;
//    const int INIT_PERIOD = 256;
//    const int MAX_PERIOD = 4096;
//    int SLEEP_PERIOD = INIT_PERIOD;
//    unsigned partsScattered = 0;
//
//    // Check queue to see if partition ready
//    while (partsScattered < numLambdasForward) {
//        consumerQueueLock.lock();
//        if (rangesToScatter.empty()) {
//            consumerQueueLock.unlock();
//            // sleep with backoff
//            usleep(SLEEP_PERIOD); // sleep a little and give up CPUs
//            failedTrials++;
//            if (failedTrials == 64 && SLEEP_PERIOD < MAX_PERIOD) {
//                failedTrials = 0;
//                SLEEP_PERIOD *= 2;
//            }
//        } else {
//            std::pair<unsigned, unsigned> partitionInfo = rangesToScatter.front();
//            rangesToScatter.pop();
//            // Has this partition already been processed
//            consumerQueueLock.unlock();
//
//            // Partition Info: (partId, rowsPerPartition)
//            unsigned startId = partitionInfo.first * partitionInfo.second;
//            unsigned endId = (partitionInfo.first + 1) * partitionInfo.second;
//            endId = endId > graph.localVtxCnt ? graph.localVtxCnt : endId;
//
//            // Create a series of buckets for batching sendout messages to nodes
//            std::vector<unsigned>* batchedIds = new std::vector<unsigned>[numNodes];
//            for (unsigned lvid = startId; lvid < endId; ++lvid) {
//                for (unsigned nid : graph.forwardGhostMap[lvid]) {
//                    batchedIds[nid].push_back(lvid);
//                }
//            }
//
//            // batch sendouts similar to the sequential version
//            bool batchFlag = true;
//            unsigned BATCH_SIZE = std::max(((batchFlag ? MAX_MSG_SIZE : 4096) - DATA_HEADER_SIZE) /
//                                           (sizeof(unsigned) + sizeof(FeatType) * featDim), 1ul); // at least send one vertex
//            for (unsigned nid = 0; nid < numNodes; ++nid) {
//                if (nid == nodeId) {
//                    continue;
//                }
//
//                unsigned forwardGhostVCnt = batchedIds[nid].size();
//                for (unsigned ib = 0; ib < forwardGhostVCnt; ib += BATCH_SIZE) {
//                    unsigned sendBatchSize = (forwardGhostVCnt - ib) < BATCH_SIZE ? (forwardGhostVCnt - ib) : BATCH_SIZE;
//
//                    forwardVerticesPushOut(nid, sendBatchSize, batchedIds[nid].data() + ib, inputTensor, featDim);
//                    fwdRecvCntLock.lock();
//                    fwdRecvCnt++;
//                    fwdRecvCntLock.unlock();
//                }
//            }
//
//            delete[] batchedIds;
//            failedTrials = 0;
//            SLEEP_PERIOD = INIT_PERIOD;
//            partsScattered++;
//        }
//    }
//
//    // Once all partitions scattered, wait on all acks
//    fwdRecvCntLock.lock();
//    if (fwdRecvCnt > 0) {
//        fwdRecvCntCond.wait();
//    }
//    fwdRecvCntLock.unlock();
}

inline void
Engine::forwardVerticesPushOut(unsigned receiver, unsigned totCnt, unsigned *lvids, FeatType *inputTensor, unsigned featDim) {
    zmq::message_t msg(DATA_HEADER_SIZE + (sizeof(unsigned) + sizeof(FeatType) * featDim) * totCnt);
    char *msgPtr = (char *)(msg.data());
    sprintf(msgPtr, NODE_ID_HEADER, receiver);
    msgPtr += NODE_ID_DIGITS;
    *(unsigned*)msgPtr = nodeId;
    msgPtr += sizeof(unsigned);
    *(unsigned*)msgPtr = totCnt;
    msgPtr += sizeof(unsigned);

    for (unsigned i = 0; i < totCnt; ++i) {
        *(unsigned *)msgPtr = graph.localToGlobalId[lvids[i]];
        msgPtr += sizeof(unsigned);
        FeatType *dataPtr = getVtxFeat(inputTensor, lvids[i], featDim);
        memcpy(msgPtr, dataPtr, sizeof(FeatType) * featDim);
        msgPtr += sizeof(FeatType) * featDim;
    }
    commManager.rawMsgPushOut(msg);
}

/**
 *
 * Major part of the engine's communication logic is done by data threads.
 * These threads loop asynchronously with computation workers.
 *
 */
void
Engine::pipelineGhostReceiver(unsigned tid) {
    // backoff sleep strategy to improve CPU utilization
    int failedTrials = 0;
    const int INIT_PERIOD = 256;
    const int MAX_PERIOD = 4096;
    int SLEEP_PERIOD = INIT_PERIOD;
    unsigned sender, topic;
    unsigned vtcsRecvd = 0;
    FeatType *msgBuf = (FeatType *)new char[MAX_MSG_SIZE];

    // While loop, looping infinitely to get the next message.
    while (!commHalt) {
        // No message in queue.
        if (!commManager.dataPullIn(&sender, &topic, msgBuf, MAX_MSG_SIZE)) {
            // Computation workers done their work, so communicator goes to death as well.
            if (commHalt) {
                delete[] msgBuf;
                if (commManager.dataPullIn(&sender, &topic, msgBuf, MAX_MSG_SIZE)) {
                    printLog(nodeId, "\033[1;31m[ ERROR ]\033[0m Still messages in buffer");
                }

                return;
            }

            usleep(SLEEP_PERIOD); // sleep a little and give up CPUs
            failedTrials++;
            if (failedTrials == 64 && SLEEP_PERIOD < MAX_PERIOD) {
                failedTrials = 0;
                SLEEP_PERIOD *= 2;
            }
            // Pull in the next message, and process this message.
        } else {
            // A normal ghost value broadcast.
            if (topic < MAX_IDTYPE - 1) {
                // Using MAX_IDTYPE - 1 as the receive signal.
                commManager.dataPushOut(sender, nodeId, MAX_IDTYPE - 1, NULL, 0);
                vtcsRecvd += topic;

                char *bufPtr = (char *)msgBuf;
                unsigned recvGhostVCnt = topic;
                unsigned featDim = getFeatDim(layer + 1);
                // Update ghost vertices
                for (unsigned i = 0; i < recvGhostVCnt; ++i) {
                    unsigned gvid = *(unsigned *)bufPtr;
                    bufPtr += sizeof(unsigned);
                    FeatType *dataPtr = getVtxFeat(forwardGhostVerticesDataOut, graph.srcGhostVtcs[gvid] - graph.localVtxCnt, featDim);
                    memcpy(dataPtr, bufPtr, sizeof(FeatType) * featDim);
                    bufPtr += sizeof(FeatType) * featDim;
                }

                // A respond to a broadcast, and the topic vertex is in my local vertices. I should update the
                // corresponding recvWaiter's value. If waiters become empty, send a signal in case the workers are
                // waiting on it to be empty at the layer barrier.
            } else { // (topic == MAX_IDTYPE - 1)
                recvCntLock.lock();
                recvCnt--;
                recvCntLock.unlock();
            }
            recvCntLock.lock();
            if (recvCnt == 0 && vtcsRecvd == graph.srcGhostCnt) {
                recvCntCond.signal();
            }
            recvCntLock.unlock();

            SLEEP_PERIOD = INIT_PERIOD;
            failedTrials = 0;
        }
    }

    if (commManager.dataPullIn(&sender, &topic, msgBuf, MAX_MSG_SIZE)) {
        printLog(nodeId, "\033[1;31m[ ERROR ]\033[0m Still messages in buffer");
    }

    delete[] msgBuf;
}

void
Engine::aggregator(unsigned tid) {
    printLog(nodeId, "AGGREGATE: Starting");
    unsigned failedTrials = 0;
    const int INIT_PERIOD = 256;
    const int MAX_PERIOD = 4096;
    int SLEEP_PERIOD = INIT_PERIOD;

    while (true) {
        aggQueueLock.lock();
        if (aggregateQueue.empty()) {
            aggQueueLock.unlock();
            usleep(SLEEP_PERIOD);
            failedTrials++;
            if (failedTrials == 64 && SLEEP_PERIOD < MAX_PERIOD) {
                failedTrials = 0;
                SLEEP_PERIOD *= 2;
            }
        } else {
            Chunk c = aggregateQueue.top();

            // There is a chunk but it is beyond the staleness bound
            if (staleness != UINT_MAX && c.layer == 0 && c.dir == PROP_TYPE::FORWARD
              && c.epoch > minEpoch + staleness) {
                aggQueueLock.unlock();
                usleep(SLEEP_PERIOD);
                failedTrials++;
                if (failedTrials == 64 && SLEEP_PERIOD < MAX_PERIOD) {
                    failedTrials = 0;
                    SLEEP_PERIOD *= 2;
                }
            // There is a chunk to process that is within the bound
            } else {
                printLog(nodeId, "AGGREGATE: Got %s", c.str().c_str());
                aggregateQueue.pop();
                aggQueueLock.unlock();

                if (c.dir == PROP_TYPE::FORWARD) {
                    aggregateChunk(c);
                } else {
                    aggregateBPChunk(c);
                }
                resComm->NNCompute(c);

                SLEEP_PERIOD = INIT_PERIOD;
                failedTrials = 0;
            }
        }
    }
}

void
Engine::scatterWorker(unsigned tid) {
    printLog(nodeId, "SCATTER: Starting");
    int failedTrials = 0;
    const int INIT_PERIOD = 256;
    const int MAX_PERIOD = 4096;
    int SLEEP_PERIOD = INIT_PERIOD;

    // Check queue to see if partition ready
    while (!commHalt) {
        scatQueueLock.lock();
        if (scatterQueue.empty()) {
            scatQueueLock.unlock();
            // sleep with backoff
            usleep(SLEEP_PERIOD); // sleep a little and give up CPUs
            failedTrials++;
            if (failedTrials == 64 && SLEEP_PERIOD < MAX_PERIOD) {
                failedTrials = 0;
                SLEEP_PERIOD *= 2;
            }
        } else {
            Chunk c = scatterQueue.top();
            scatterQueue.pop();
            scatQueueLock.unlock();

            printLog(nodeId, "SCATTER: Got %s", c.str().c_str());

            // Get the layer output you want to scatter
            // If forward then it was the previous layer output
            // Need featLayer because apparently the featDim differs
            //  from the outputLayer depending on forward or backward
            //  TODO: Definitely implement the only ascending version
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

            FeatType* scatterTensor = savedNNTensors[outputLayer][tensorName].getData();

            unsigned startId = c.lowBound;
            unsigned endId = c.upBound;
            unsigned featDim = getFeatDim(featLayer);

            PROP_TYPE dir = c.dir;
            std::map<unsigned, std::vector<unsigned>>& ghostMap = dir == PROP_TYPE::FORWARD ?
              graph.forwardGhostMap : graph.backwardGhostMap;

            // Create a series of buckets for batching sendout messages to nodes
            std::vector<unsigned>* batchedIds = new std::vector<unsigned>[numNodes];
            for (unsigned lvid = startId; lvid < endId; ++lvid) {
                for (unsigned nid : ghostMap[lvid]) {
                    batchedIds[nid].push_back(lvid);
                }
            }

            // batch sendouts similar to the sequential version
            bool batchFlag = true;
            unsigned BATCH_SIZE = std::max(((batchFlag ? MAX_MSG_SIZE : 4096) - DATA_HEADER_SIZE) /
                                           (sizeof(unsigned) + sizeof(FeatType) * featDim), 1ul); // at least send one vertex
            for (unsigned nid = 0; nid < numNodes; ++nid) {
                if (nid == nodeId) {
                    continue;
                }

                unsigned ghostVCnt = batchedIds[nid].size();
                for (unsigned ib = 0; ib < ghostVCnt; ib += BATCH_SIZE) {
                    unsigned sendBatchSize = (ghostVCnt - ib) < BATCH_SIZE ? (ghostVCnt - ib) : BATCH_SIZE;

                    verticesPushOut(nid, sendBatchSize, batchedIds[nid].data() + ib, scatterTensor, featDim, c);
                    recvCntLock.lock();
                    recvCnt++;
                    recvCntLock.unlock();
                }

                // Once all partitions scattered, wait on all acks
                recvCntLock.lock();
                if (recvCnt > 0) {
                    recvCntCond.wait();
                }
                recvCntLock.unlock();

            }
            // Add chunk into appropriate aggregate queue
            printLog(nodeId, "SCATTER: Finished %s", c.str().c_str());
            aggregateQueue.push(c);

            delete[] batchedIds;
            failedTrials = 0;
            SLEEP_PERIOD = INIT_PERIOD;
        }
    }
}


void
Engine::verticesPushOut(unsigned receiver, unsigned totCnt, unsigned *lvids, FeatType *inputTensor, unsigned featDim, Chunk& c) {
    zmq::message_t msg(DATA_HEADER_SIZE + (sizeof(unsigned) + sizeof(FeatType) * featDim) * totCnt);
    char *msgPtr = (char *)(msg.data());
    sprintf(msgPtr, NODE_ID_HEADER, receiver);
    msgPtr += NODE_ID_DIGITS;
    populateHeader(msgPtr, nodeId, totCnt, featDim, c.layer, c.dir);
    msgPtr += sizeof(unsigned) * 5;

    for (unsigned i = 0; i < totCnt; ++i) {
        *(unsigned *)msgPtr = graph.localToGlobalId[lvids[i]];
        msgPtr += sizeof(unsigned);
        FeatType *dataPtr = getVtxFeat(inputTensor, lvids[i], featDim);
        memcpy(msgPtr, dataPtr, sizeof(FeatType) * featDim);
        msgPtr += sizeof(FeatType) * featDim;
    }
    commManager.rawMsgPushOut(msg);
}


void
Engine::ghostReceiver(unsigned tid) {
    printLog(nodeId, "RECEIVER: Starting");
    // backoff sleep strategy to improve CPU utilization
    int failedTrials = 0;
    const int INIT_PERIOD = 256;
    const int MAX_PERIOD = 4096;
    int SLEEP_PERIOD = INIT_PERIOD;
    unsigned sender, topic;
    std::string tensorName;
    FeatType *msgBuf = (FeatType *)new char[MAX_MSG_SIZE];

    // While loop, looping infinitely to get the next message.
    while (!commHalt) {
        // No message in queue.
        if (!commManager.dataPullIn(&sender, &topic, msgBuf, MAX_MSG_SIZE)) {
            // Computation workers done their work, so communicator goes to death as well.
            if (commHalt) {
                delete[] msgBuf;
                if (commManager.dataPullIn(&sender, &topic, msgBuf, MAX_MSG_SIZE)) {
                    printLog(nodeId, "\033[1;31m[ ERROR ]\033[0m Still messages in buffer");
                }

                return;
            }

            usleep(SLEEP_PERIOD); // sleep a little and give up CPUs
            failedTrials++;
            if (failedTrials == 64 && SLEEP_PERIOD < MAX_PERIOD) {
                failedTrials = 0;
                SLEEP_PERIOD *= 2;
            }
            // Pull in the next message, and process this message.
        } else {
            // A normal ghost value broadcast.
            if (topic < MAX_IDTYPE - 1) {
                // Using MAX_IDTYPE - 1 as the receive signal.
                commManager.dataPushOut(sender, nodeId, MAX_IDTYPE - 1, NULL, 0);

                char *bufPtr = (char *)msgBuf;
                unsigned recvGhostVCnt = topic;
                unsigned featDim = *(unsigned *)bufPtr;
                bufPtr += sizeof(unsigned);
                unsigned layer = *(unsigned *)bufPtr;
                bufPtr += sizeof(unsigned);
                unsigned dir = *(unsigned *)bufPtr;
                bufPtr += sizeof(unsigned);
                // Get proper variables depending on forward or backward
                if (dir == PROP_TYPE::FORWARD) {
                    tensorName = "fg";
                } else {
                    tensorName = "bg";
                }
                std::map<unsigned, unsigned>& globalToGhostVtcs = dir == PROP_TYPE::FORWARD
                  ? graph.srcGhostVtcs : graph.dstGhostVtcs;

                printLog(nodeId, "RECEIVER: Got msg %u:%s", layer,
                  dir == PROP_TYPE::FORWARD ? "F" : "B");
                FeatType* ghostData = savedNNTensors[layer][tensorName].getData();
                if (ghostData == NULL) {
                    printLog(nodeId, "RECEIVER: Coudn't find tensor '%s' for layer %u",
                      tensorName.c_str(), layer);
                }

                // Update ghost vertices
                for (unsigned i = 0; i < recvGhostVCnt; ++i) {
                    unsigned gvid = *(unsigned *)bufPtr;
                    bufPtr += sizeof(unsigned);
                    FeatType *dataPtr = getVtxFeat(ghostData, globalToGhostVtcs[gvid] - graph.localVtxCnt, featDim);
                    memcpy(dataPtr, bufPtr, sizeof(FeatType) * featDim);
                    bufPtr += sizeof(FeatType) * featDim;
                }

                // A respond to a broadcast, and the topic vertex is in my local vertices. I should update the
                // corresponding recvWaiter's value. If waiters become empty, send a signal in case the workers are
                // waiting on it to be empty at the layer barrier.
            } else { // (topic == MAX_IDTYPE - 1)
                recvCntLock.lock();
                recvCnt--;
                recvCntLock.unlock();
            }
            recvCntLock.lock();
            if (recvCnt == 0) {
                recvCntCond.signal();
            }
            recvCntLock.unlock();

            SLEEP_PERIOD = INIT_PERIOD;
            failedTrials = 0;
        }
    }

    if (commManager.dataPullIn(&sender, &topic, msgBuf, MAX_MSG_SIZE)) {
        printLog(nodeId, "\033[1;31m[ ERROR ]\033[0m Still messages in buffer");
    }

    delete[] msgBuf;
}


/**
 *
 * Major part of the engine's communication logic is done by data threads.
 * These threads loop asynchronously with computation workers.
 *
 */
void
Engine::forwardGhostReceiver(unsigned tid) {
    // backoff sleep strategy to improve CPU utilization
    int failedTrials = 0;
    const int INIT_PERIOD = 256;
    const int MAX_PERIOD = 4096;
    int SLEEP_PERIOD = INIT_PERIOD;
    unsigned sender, topic;
    unsigned vtcsRecvd = 0;
    unsigned featDim = getFeatDim(layer + 1);
    FeatType *msgBuf = (FeatType *)new char[MAX_MSG_SIZE];

    // While loop, looping infinitely to get the next message.
    while (!commHalt) {
        // No message in queue.
        if (!commManager.dataPullIn(&sender, &topic, msgBuf, MAX_MSG_SIZE)) {
            // Computation workers done their work, so communicator goes to death as well.
            if (commHalt) {
                delete[] msgBuf;
                if (commManager.dataPullIn(&sender, &topic, msgBuf, MAX_MSG_SIZE)) {
                    printLog(nodeId, "\033[1;31m[ ERROR ]\033[0m Still messages in buffer");
                }

                return;
            }

            usleep(SLEEP_PERIOD); // sleep a little and give up CPUs
            failedTrials++;
            if (failedTrials == 64 && SLEEP_PERIOD < MAX_PERIOD) {
                failedTrials = 0;
                SLEEP_PERIOD *= 2;
            }
            // Pull in the next message, and process this message.
        } else {
            // A normal ghost value broadcast.
            if (topic < MAX_IDTYPE - 1) {
                // Using MAX_IDTYPE - 1 as the receive signal.
                commManager.dataPushOut(sender, nodeId, MAX_IDTYPE - 1, NULL, 0);
                vtcsRecvd += topic;

                char *bufPtr = (char *)msgBuf;
                unsigned recvGhostVCnt = topic;
                // Update ghost vertices
                for (unsigned i = 0; i < recvGhostVCnt; ++i) {
                    unsigned gvid = *(unsigned *)bufPtr;
                    bufPtr += sizeof(unsigned);
                    FeatType *dataPtr = getVtxFeat(forwardGhostVerticesDataOut, graph.srcGhostVtcs[gvid] - graph.localVtxCnt, featDim);
                    memcpy(dataPtr, bufPtr, sizeof(FeatType) * featDim);
                    bufPtr += sizeof(FeatType) * featDim;
                }

                // A respond to a broadcast, and the topic vertex is in my local vertices. I should update the
                // corresponding recvWaiter's value. If waiters become empty, send a signal in case the workers are
                // waiting on it to be empty at the layer barrier.
            } else { // (topic == MAX_IDTYPE - 1)
                recvCntLock.lock();
                recvCnt--;
                recvCntLock.unlock();
            }
            recvCntLock.lock();
            if (recvCnt == 0 && vtcsRecvd == graph.srcGhostCnt) {
                recvCntCond.signal();
            }
            recvCntLock.unlock();

            SLEEP_PERIOD = INIT_PERIOD;
            failedTrials = 0;
        }
    }

    if (commManager.dataPullIn(&sender, &topic, msgBuf, MAX_MSG_SIZE)) {
        printLog(nodeId, "\033[1;31m[ ERROR ]\033[0m Still messages in buffer");
    }

    delete[] msgBuf;
}

// reshape vtcs tensor to edgs tensor. Each element in edgsTensor is a reference to a vertex feature.
// Both src vtx features and dst vtx features included in edgsTensor. [srcV Feats (local inEdge cnt); dstV Feats (local inEdge cnt)]
FeatType **
Engine::srcVFeats2eFeats(FeatType *vtcsTensor, FeatType* ghostTensor, unsigned vtcsCnt, unsigned featDim) {
    underlyingVtcsTensorBuf = vtcsTensor;

    FeatType** eVtxFeatsBuf = new FeatType *[2 * graph.localInEdgeCnt];
    FeatType **eSrcVtxFeats = eVtxFeatsBuf;
    FeatType **eDstVtxFeats = eSrcVtxFeats + graph.localInEdgeCnt;

    unsigned long long edgeItr = 0;
    for (unsigned lvid = 0; lvid < graph.localVtxCnt; ++lvid) {
        for (unsigned long long eid = graph.forwardAdj.columnPtrs[lvid]; eid < graph.forwardAdj.columnPtrs[lvid + 1]; ++eid) {
            unsigned srcVid = graph.forwardAdj.rowIdxs[eid];
            if (srcVid < graph.localVtxCnt) {
                eSrcVtxFeats[edgeItr] = getVtxFeat(vtcsTensor, srcVid, featDim);
            } else {
                eSrcVtxFeats[edgeItr] = getVtxFeat(ghostTensor, srcVid - graph.localVtxCnt, featDim);
            }
            eDstVtxFeats[edgeItr] = getVtxFeat(vtcsTensor, lvid, featDim);
            ++edgeItr;
        }
    }

    return eVtxFeatsBuf;
}

// similar to srcVFeats2eFeats, but based on outEdges of local vertices.
// [dstV Feats (local outEdge cnt); srcV Feats (local outEdge cnt)]
FeatType **
Engine::dstVFeats2eFeats(FeatType *vtcsTensor, FeatType* ghostTensor, unsigned vtcsCnt, unsigned featDim) {
    underlyingVtcsTensorBuf = vtcsTensor;

    FeatType **eVtxFeatsBuf = new FeatType *[2 * graph.localOutEdgeCnt];
    FeatType **eSrcVtxFeats = eVtxFeatsBuf;
    FeatType **eDstVtxFeats = eSrcVtxFeats + graph.localOutEdgeCnt;

    unsigned long long edgeItr = 0;
    for (unsigned lvid = 0; lvid < graph.localVtxCnt; ++lvid) {
        for (unsigned long long eid = graph.backwardAdj.rowPtrs[lvid]; eid < graph.backwardAdj.rowPtrs[lvid + 1]; ++eid) {
            unsigned srcVid = graph.backwardAdj.columnIdxs[eid];
            if (srcVid < graph.localVtxCnt) {
                eSrcVtxFeats[edgeItr] = getVtxFeat(vtcsTensor, srcVid, featDim);
            } else {
                eSrcVtxFeats[edgeItr] = getVtxFeat(ghostTensor, srcVid - graph.localVtxCnt, featDim);
            }
            eDstVtxFeats[edgeItr] = getVtxFeat(vtcsTensor, lvid, featDim);
            ++edgeItr;
        }
    }

    return eVtxFeatsBuf;
}

// FeatType *
// Engine::eFeats2srcVFeats(FeatType **edgsTensor, unsigned edgsCnt, unsigned featDim) {
//     FeatType *vtcsTensor = new FeatType [graph.localVtxCnt];
//     memset(vtcsTensor, 0, sizeof(FeatType) * graph.localVtxCnt);

//     unsigned edgeItr = 0;
//     for (unsigned lvid = 0; lvid < graph.localVtxCnt; ++lvid) {
//         FeatType *vtxFeat = getVtxFeat(vtcsTensor, lvid, featDim);
//         for (unsigned long long eid = graph.forwardAdj.columnPtrs[lvid]; eid < graph.forwardAdj.columnPtrs[lvid + 1]; ++eid) {
//             for (unsigned j = 0; j < featDim; ++j) {
//                 vtxFeat[j] += edgsTensor[edgeItr][j];
//             }
//             ++edgeItr;
//         }
//     }

//     return vtcsTensor;
// }

// FeatType *
// Engine::eFeats2dstVFeats(FeatType **edgsTensor, unsigned edgsCnt, unsigned featDim) {
//     FeatType *vtcsTensor = new FeatType [graph.localVtxCnt];
//     memset(vtcsTensor, 0, sizeof(FeatType) * graph.localVtxCnt);

//     unsigned edgeItr = 0;
//     for (unsigned lvid = 0; lvid < graph.localVtxCnt; ++lvid) {
//         FeatType *vtxFeat = getVtxFeat(vtcsTensor, lvid, featDim);
//         for (unsigned long long eid = graph.backwardAdj.rowPtrs[lvid]; eid < graph.backwardAdj.rowPtrs[lvid + 1]; ++eid) {
//             for (unsigned j = 0; j < featDim; ++j) {
//                 vtxFeat[j] += edgsTensor[edgeItr][j];
//             }
//             ++edgeItr;
//         }
//     }

//     return vtcsTensor;
// }


/**
 *
 * Major part of the engine's backward-prop logic.
 *
 */
#ifdef _GPU_ENABLED_
FeatType *
Engine::aggregateBackward(FeatType *gradTensor, unsigned vtcsCnt, unsigned featDim) {
    auto t0 = gtimers.getTimer("Memcpy2GPUBackwardTimer");
    auto t1 = gtimers.getTimer("AggBackwardTimer");
    auto t2 = gtimers.getTimer("ComputeTransBackwardTimer");
    auto t3 = gtimers.getTimer("Memcpy2RAMBackwardTimer");
    double sttTimer = getTimer();
    currId = 0;
    FeatType *outputTensor = new FeatType[vtcsCnt * featDim];
    CuMatrix feat;
    t0->start();
    feat.loadSpDense(gradTensor, backwardGhostVerticesDataIn,
                     graph.localVtxCnt, graph.dstGhostCnt,
                     featDim);
    cudaDeviceSynchronize();
    t0->stop();
    t1->start();
    CuMatrix out = cu.aggregate(*NormAdjMatrixOut, feat);
    cudaDeviceSynchronize();
    t1->stop();
    t2->start();
    out = out.transpose();
    cudaDeviceSynchronize();
    t2->stop();
    t3->start();
    out.setData(outputTensor);
    out.updateMatrixFromGPU();
    t3->stop();

    currId = vtcsCnt;

    delete[] gradTensor;
    delete[] backwardGhostVerticesDataIn;

    if (vecTimeAggregate.size() < 2 * numLayers) {
        for (unsigned i = vecTimeAggregate.size(); i < 2 * numLayers; i++) {
            vecTimeAggregate.push_back(0.0);
        }
    }
    vecTimeAggregate[numLayers + layer] += getTimer() - sttTimer;

    return outputTensor;
}
#else
FeatType*
Engine::aggregateBackward(FeatType **eVGradTensor, unsigned edgsCnt, unsigned featDim, AGGREGATOR aggregator) {
    double sttTimer = getTimer();

    FeatType* outputTensor = savedNNTensors[layer-1]["aTg"].getData();
    FeatType* gradTensor = savedNNTensors[layer]["grad"].getData();
    currId = 0;

    switch (aggregator) {
        case (AGGREGATOR::WSUM): {
            memcpy(outputTensor, gradTensor, sizeof(FeatType) * graph.localVtxCnt * featDim);

            AggOPArgs args = {outputTensor, eVGradTensor, graph.localVtxCnt, edgsCnt, featDim};
            auto computeFn = std::bind(&Engine::aggregateBPCompute, this, std::placeholders::_1, std::placeholders::_2);
            computePool->perform(computeFn, &args);
            computePool->sync();
            break;
        }
        default:
            printLog(nodeId, "Invalid aggregator %d", aggregator);
            break;
    }

    if (vecTimeAggregate.size() < 2 * numLayers) {
        for (unsigned i = vecTimeAggregate.size(); i < 2 * numLayers; i++) {
            vecTimeAggregate.push_back(0.0);
        }
    }
    vecTimeAggregate[numLayers + layer] += getTimer() - sttTimer;

    return outputTensor;
}
#endif



FeatType *
Engine::applyVertexBackward(FeatType *gradTensor, unsigned vtcsCnt, unsigned inFeatDim, unsigned outFeatDim) {
    double sttTimer = getTimer();

    assert(vtcsCnt == graph.localVtxCnt);
    FeatType* outputTensor = savedNNTensors[layer-1]["grad"].getData();

    if (vecTimeLambdaInvoke.size() < 2 * numLayers) {
        for (unsigned i = vecTimeLambdaInvoke.size(); i < 2 * numLayers; ++i) {
            vecTimeLambdaInvoke.push_back(0.0);
            vecTimeLambdaWait.push_back(0.0);
        }
    }

    if (mode == LAMBDA) {
        for (unsigned u = 0; u < numLambdasForward; ++u) {
            unsigned chunkSize = (vtcsCnt + numLambdasForward - 1) / numLambdasForward;
            unsigned lowBound = u * chunkSize;
            unsigned upBound = std::min(lowBound + chunkSize, vtcsCnt);
            Chunk chunk {u, lowBound, upBound, layer-1, PROP_TYPE::BACKWARD, currEpoch, true}; // epoch doesn't matter in sync version
            resComm->NNCompute(chunk);
        }
        resComm->NNSync();
    } else if (mode == GPU) { // TODO: (YIFAN) support for GPU/CPU
    } else if (mode == CPU) {
    }

    if (vecTimeApplyVtx.size() < 2 * numLayers) {
        for (unsigned i = vecTimeApplyVtx.size(); i < 2 * numLayers; i++) {
            vecTimeApplyVtx.push_back(0.0);
        }
    }
    vecTimeApplyVtx[numLayers + layer - 1] += getTimer() - sttTimer;
    return outputTensor;
}


FeatType **
Engine::applyEdgeBackward(EdgeType *edgsTensor, unsigned edgsCnt, unsigned eFeatDim, FeatType **eSrcVGradTensor, FeatType **eDstVGradTensor, unsigned inFeatDim, unsigned outFeatDim) {
    double sttTimer = getTimer();

    FeatType **outputTensor = eDstVGradTensor;
    eDstVGradTensor = NULL;

    for (auto &sTensor: edgNNSavedTensors[layer - 1]) {
        delete[] sTensor.getData();
    }
    if (vecTimeApplyEdg.size() < 2 * numLayers) {
        for (unsigned i = vecTimeApplyEdg.size(); i < 2 * numLayers; i++) {
            vecTimeApplyEdg.push_back(0.0);
        }
    }
    vecTimeApplyEdg[numLayers + layer - 1] += getTimer() - sttTimer;
    return outputTensor;
}

FeatType **
Engine::scatterBackward(FeatType *gradTensor, unsigned vtcsCnt, unsigned featDim) {
    double sttTimer = getTimer();

    // Start data communicators.
    commHalt = false;
    recvCnt = 0;
    // YIFAN: Do we really need reset bkwdRecvCnt? Same question for all 3 reset
    // JOHN: Yifan, no we don't. I implemented that when I suspected that
    //  having a single counter for fwd and bkwd was the problem with pipelining
    backwardGhostVerticesDataOut = savedNNTensors[layer-1]["bg"].getData();
    auto bgr_fp = std::bind(&Engine::backwardGhostReceiver, this,
      std::placeholders::_1);
    dataPool->perform(bgr_fp);

    sendBackwardGhostGradients(gradTensor, featDim);

    //## Global Iteration barrier. ##/
    // TODO: (YIFAN) we can optimize this to extend comm protocal. Mark the last packet sent so this node knows when to exit ghostCommunicator.
    nodeManager.barrier();
    commHalt = true;
    // Join all data communicators.
    dataPool->sync();

    //FeatType **eFeats = dstVFeats2eFeats(gradTensor, backwardGhostVerticesDataIn, vtcsCnt, featDim);
    FeatType **eFeats = savedEdgeTensors[layer-1]["bedge"];
    gradTensor = NULL;

    if (vecTimeScatter.size() < 2 * numLayers) {
        for (unsigned i = vecTimeScatter.size(); i < 2 * numLayers; i++) {
            vecTimeScatter.push_back(0.0);
        }
    }
    vecTimeScatter[numLayers + layer - 1] += getTimer() - sttTimer;
    return eFeats;
}

FeatType*
Engine::fusedGASBackward(FeatType *gradTensor, unsigned vtcsCnt, unsigned inFeatDim, unsigned outFeatDim, bool aggregate, bool scatter) {
    return NULL;
//    double sttTimer = getTimer();
//
//    consumerQueueLock.lock();
//    while (!rangesToScatter.empty()) rangesToScatter.pop();
//    consumerQueueLock.unlock();
//
//    // Case 1 - First phase, no aggregate needed
//    FeatType* outputTensor = nullptr;
//    if (!aggregate && scatter) {
//        outputTensor = applyScatterPhase(gradTensor, vtcsCnt, inFeatDim, outFeatDim, scatter);
//    }
//    // Case 2 - Full phase including gather, apply, and scatter
//    else if (aggregate && scatter) {
//        outputTensor = aggregateApplyScatterPhase(gradTensor, vtcsCnt, inFeatDim, outFeatDim, scatter);
//    }
//    // Case 3 - Final phase, no scatter needed
//    else if (aggregate && !scatter) {
//        outputTensor = aggregateApplyPhase(gradTensor, vtcsCnt, inFeatDim, outFeatDim, scatter);
//    }
//    else {
//        printLog(nodeId, "\033[1;33m[ UNKOWN ]\033[0m No scatter or aggregate phase");
//    }
//
//    if (vecTimeAggregate.size() < 2 * numLayers) {
//        for (unsigned i = vecTimeAggregate.size(); i < 2 * numLayers; i++) {
//            vecTimeAggregate.push_back(0.0);
//        }
//    }
//    vecTimeAggregate[numLayers + layer - 1] += getTimer() - sttTimer;
//
//    backwardGhostVerticesDataIn = backwardGhostVerticesDataOut;
//
//    return outputTensor;
}


//////////////////////////////////////////////////////////
// Below are private backward functions for the engine. //
//////////////////////////////////////////////////////////
// Backward scatter phase functions
FeatType* Engine::applyScatterPhase(FeatType* gradTensor, unsigned vtcsCnt,
  unsigned inFeatDim, unsigned outFeatDim, bool scatter) {
    return NULL;
    // double sttTimer = getTimer();

    // assert(vtcsCnt == graph.localVtxCnt);
    // commHalt = false;
    // bkwdRecvCnt = 0;

    // FeatType *outputTensor = new FeatType[vtcsCnt * inFeatDim];
    // auto bgr_fp = std::bind(&Engine::backwardGhostReceiver, this,
    //                 std::placeholders::_1, std::placeholders::_2);
    // auto bgu_fp = std::bind(&Engine::pipelineBackwardGhostGradients, this,
    //                 std::placeholders::_1, std::placeholders::_2);
    // std::thread scatterThread;
    // if (scatter) {
    //     backwardGhostVerticesDataOut = new FeatType[graph.dstGhostCnt * inFeatDim];
    //     dataPool->perform(bgr_fp, (void*) &inFeatDim);
    //     scatterThread = std::thread(bgu_fp, outputTensor, inFeatDim);
    // }

    // Matrix inputTensor_ = Matrix(vtcsCnt, outFeatDim, gradTensor);
    // Matrix outputTensor_ = Matrix(vtcsCnt, inFeatDim, outputTensor);
    // Matrix targetTensor_ = Matrix(vtcsCnt, getFeatDim(numLayers), localVerticesLabels);
    // resComm->newContext(iteration - 1, inputTensor_, outputTensor_, targetTensor_,
    //                     vtxNNSavedTensors, scatter);
    // resComm->requestBackward(iteration - 1, iteration - 1 == numLayers - 1);

    // if (scatter) {
    //     scatterThread.join();
    // }
    // commHalt = true;
    // dataPool->sync();

    // delete[] gradTensor;
    // for (auto &sTensor : vtxNNSavedTensors[iteration - 1]) {
    //     delete[] sTensor.getData();
    // }
    // vtxNNSavedTensors[iteration - 1].clear();

    // if (vecTimeApplyVtx.size() < 2 * numLayers) {
    //     for (unsigned i = vecTimeApplyVtx.size(); i < 2 * numLayers; i++) {
    //         vecTimeApplyVtx.push_back(0.0);
    //     }
    // }
    // vecTimeApplyVtx[numLayers + iteration - 1] += getTimer() - sttTimer;

    // return outputTensor;
}

FeatType* Engine::aggregateApplyScatterPhase(FeatType* gradTensor, unsigned vtcsCnt,
  unsigned inFeatDim, unsigned outFeatDim, bool scatter) {
    return NULL;
    // // Prepare for gather phase
    // FeatType *gatheredTensor = new FeatType[vtcsCnt * outFeatDim];
    // FeatType *outputTensor = new FeatType[vtcsCnt * inFeatDim];
    // auto bgr_fp = std::bind(&Engine::backwardGhostReceiver, this,
    //                 std::placeholders::_1, std::placeholders::_2);
    // auto bgu_fp = std::bind(&Engine::pipelineBackwardGhostGradients, this,
    //                 std::placeholders::_1, std::placeholders::_2);
    // commHalt = false;
    // bkwdRecvCnt = 0;
    // std::thread scatterThread;
    // if (scatter) {
    //     backwardGhostVerticesDataOut = new FeatType[graph.dstGhostCnt * outFeatDim];
    //     dataPool->perform(bgr_fp, (void*) &inFeatDim);
    //     scatterThread = std::thread(bgu_fp, outputTensor, outFeatDim);
    // }

    // currId = 0;

    // // Start gathering
    // AggOPArgs args = {gatheredTensor, gradTensor, vtcsCnt, outFeatDim};
    // auto computeFn = std::bind(&Engine::aggregateBPCompute, this, std::placeholders::_1, std::placeholders::_2);
    // computePool->perform(computeFn, &args);

    // // Prepare for applyVertex phase
    // Matrix inputTensor_ = Matrix(vtcsCnt, outFeatDim, gatheredTensor);
    // Matrix outputTensor_ = Matrix(vtcsCnt, inFeatDim, outputTensor);
    // Matrix targetTensor_ = Matrix(vtcsCnt, getFeatDim(numLayers), localVerticesLabels);
    // resComm->newContext(layer - 1, inputTensor_, outputTensor_, targetTensor_,
    //                     vtxNNSavedTensors, scatter);

    // // Start applyVertex phase
    // unsigned currLambdaId = 0;
    // if (mode == LAMBDA) {
    //     const unsigned lambdaChunkSize = (vtcsCnt + numLambdasForward - 1) / numLambdasBackward;
    //     unsigned availChunkSize = lambdaChunkSize;
    //     while (currId < vtcsCnt) {
    //         unsigned lvid = currId;
    //         if (lvid > availChunkSize) {
    //             resComm->applyVertexBackward(layer - 1, currLambdaId, layer - 1 == numLayers - 1);
    //             availChunkSize += lambdaChunkSize;
    //             ++currLambdaId;
    //         }
    //         usleep(2000); // wait for 2ms and check again
    //     }
    // }
    // computePool->sync();
    // if (mode != LAMBDA) {
    //     resComm->requestBackward(layer - 1, layer - 1 == numLayers - 1);
    // } else {
    //     while (currLambdaId < numLambdasBackward) {
    //         resComm->applyVertexBackward(layer - 1, currLambdaId, layer - 1 == numLayers - 1);
    //         ++currLambdaId;
    //     }
    //     resComm->waitResBackward(layer - 1, layer - 1 == numLayers - 1);
    // }

    // if (scatter) {
    //     scatterThread.join();
    // }
    // commHalt = true;
    // dataPool->sync();

    // // Clean up applyVertex phase
    // delete[] gatheredTensor;
    // for (auto &sTensor : vtxNNSavedTensors[layer - 1]) {
    //     delete[] sTensor.getData();
    // }
    // vtxNNSavedTensors[layer - 1].clear();

    // // Clean up gather phase
    // delete[] gradTensor;
    // delete[] backwardGhostVerticesDataIn;

    // return outputTensor;
}

FeatType* Engine::aggregateApplyPhase(FeatType* gradTensor, unsigned vtcsCnt,
  unsigned inFeatDim, unsigned outFeatDim, bool scatter) {
    return NULL;
    // double sttTimer = getTimer();

    // // Prepare for gather phase
    // FeatType *gatheredTensor = new FeatType[vtcsCnt * outFeatDim];
    // FeatType *outputTensor = new FeatType[vtcsCnt * inFeatDim];
    // currId = 0;

    // // Start gathering
    // AggOPArgs args = {gatheredTensor, gradTensor, vtcsCnt, outFeatDim};
    // auto computeFn = std::bind(&Engine::aggregateBPCompute, this, std::placeholders::_1, std::placeholders::_2);
    // computePool->perform(computeFn, &args);

    // // Prepare for applyVertex phase
    // Matrix inputTensor_ = Matrix(vtcsCnt, outFeatDim, gatheredTensor);
    // Matrix outputTensor_ = Matrix(vtcsCnt, inFeatDim, outputTensor);
    // Matrix targetTensor_ = Matrix(vtcsCnt, getFeatDim(numLayers), localVerticesLabels);
    // resComm->newContext(layer - 1, inputTensor_, outputTensor_, targetTensor_,
    //                     vtxNNSavedTensors, scatter);

    // // Start applyVertex phase
    // unsigned currLambdaId = 0;
    // if (mode == LAMBDA) {
    //     const unsigned lambdaChunkSize = (vtcsCnt + numLambdasForward - 1) / numLambdasBackward;
    //     unsigned availChunkSize = lambdaChunkSize;
    //     while (currId < vtcsCnt) {
    //         unsigned lvid = currId;
    //         if (lvid > availChunkSize) {
    //             resComm->applyVertexBackward(layer - 1, currLambdaId, layer - 1 == numLayers - 1);
    //             availChunkSize += lambdaChunkSize;
    //             ++currLambdaId;
    //         }
    //         usleep(2000); // wait for 2ms and check again
    //     }
    // }
    // computePool->sync();
    // if (mode != LAMBDA) {
    //     resComm->requestBackward(layer - 1, layer - 1 == numLayers - 1);
    // } else {
    //     while (currLambdaId < numLambdasBackward) {
    //         resComm->applyVertexBackward(layer - 1, currLambdaId, layer - 1 == numLayers - 1);
    //         ++currLambdaId;
    //     }
    //     resComm->waitResBackward(layer - 1, layer - 1 == numLayers - 1);
    // }

    // // Clean up applyVertex phase
    // delete[] gatheredTensor;
    // for (auto &sTensor : vtxNNSavedTensors[layer - 1]) {
    //     delete[] sTensor.getData();
    // }
    // vtxNNSavedTensors[layer - 1].clear();

    // // Clean up gather phase
    // delete[] gradTensor;
    // delete[] backwardGhostVerticesDataIn;

    // if (vecTimeAggregate.size() < 2 * numLayers) {
    //     for (unsigned i = vecTimeAggregate.size(); i < 2 * numLayers; i++) {
    //         vecTimeAggregate.push_back(0.0);
    //     }
    // }
    // vecTimeAggregate[numLayers + layer] += getTimer() - sttTimer;

    // return outputTensor;
}

void Engine::aggregateBPCompute(unsigned tid, void *args) {
    FeatType *nextGradTensor = ((AggOPArgs *) args)->outputTensor;
    FeatType **gradTensor = ((AggOPArgs *) args)->inputTensor;
    const unsigned vtcsCnt = ((AggOPArgs *) args)->vtcsCnt;
    // const unsigned edgsCnt = ((AggOPArgs *) args)->edgsCnt;
    const unsigned featDim = ((AggOPArgs *) args)->featDim;

    unsigned lvid = 0;
    while (currId < vtcsCnt) {
        lvid = __sync_fetch_and_add(&currId, 1);
        if (lvid < vtcsCnt) {
            backwardAggregateFromNeighbors(lvid, nextGradTensor, gradTensor, featDim);
        }
    }
}


void
Engine::aggregateBPChunk(Chunk& c) {
    unsigned lvid = c.lowBound;
    unsigned limit = c.upBound;
    unsigned featDim = getFeatDim(c.layer + 1);

    FeatType* featTensor = getVtxFeat(savedNNTensors[c.layer + 1]["grad"].getData(),
      lvid, featDim);
    FeatType* aggTensor = savedNNTensors[c.layer]["aTg"].getData();
    FeatType** eFeatsTensor = savedEdgeTensors[c.layer]["bedge"];

    FeatType* chunkPtr = getVtxFeat(aggTensor, lvid, featDim);
    std::memcpy(chunkPtr, featTensor, sizeof(FeatType) * (limit - lvid) * featDim);
    while (lvid < limit) {
        backwardAggregateFromNeighbors(lvid++, aggTensor, eFeatsTensor, featDim);
    }
}


/**
 *
 * Aggregate featDim feature values starting from offset from all neighbors (including self). Then write the results to the
 * data buffer area for serialization. The results are to be used for being sent to lambda threads.
 *
 */
void
Engine::backwardAggregateFromNeighbors(unsigned lvid, FeatType *nextGradTensor, FeatType **gradTensor, unsigned featDim) {
    // Read out data of the current layer of given vertex.
    FeatType *currDataDst = getVtxFeat(nextGradTensor, lvid, featDim);

    // Apply normalization factor on the current data.
    {
        const EdgeType normFactor = graph.vtxDataVec[lvid];
        for (unsigned i = 0; i < featDim; ++i) {
            currDataDst[i] *= normFactor;
        }
    }

    // Aggregate from neighbors.
    for (unsigned long long eid = graph.backwardAdj.rowPtrs[lvid]; eid < graph.backwardAdj.rowPtrs[lvid + 1]; ++eid) {
        EdgeType normFactor = graph.backwardAdj.values[eid];
        for (unsigned j = 0; j < featDim; ++j) {
            currDataDst[j] += gradTensor[eid][j] * normFactor;
        }
    }
}

void
Engine::sendBackwardGhostGradients(FeatType *gradTensor, unsigned featDim) {
    // Loop through all local vertices and do the data send out work. If there are any remote edges for a vertex, should send this vid to
    // other nodes for their ghost's update.
    bool batchFlag = true;
    unsigned BATCH_SIZE = std::max(((batchFlag ? MAX_MSG_SIZE : 4096) - DATA_HEADER_SIZE) /
                                   (sizeof(unsigned) + sizeof(FeatType) * featDim), 1ul); // at least send one vertex
    for (unsigned nid = 0; nid < numNodes; ++nid) {
        if (nid == nodeId) {
            continue;
        }
        unsigned backwardGhostVCnt = graph.backwardLocalVtxDsts[nid].size();
        for (unsigned ib = 0; ib < backwardGhostVCnt; ib += BATCH_SIZE) {
            unsigned sendBatchSize = (backwardGhostVCnt - ib) < BATCH_SIZE ? (backwardGhostVCnt - ib) : BATCH_SIZE;

            backwardVerticesPushOut(nid, sendBatchSize, graph.backwardLocalVtxDsts[nid].data() + ib, gradTensor, featDim);
            recvCntLock.lock();
            recvCnt++;
            recvCntLock.unlock();
        }
    }
    // Wait for all remote schedulings sent by me to be handled.
    recvCntLock.lock();
    if (recvCnt > 0) {
        recvCntCond.wait();
    }
    recvCntLock.unlock();
}

inline void
Engine::pipelineBackwardGhostGradients(FeatType* inputTensor, unsigned featDim) {
//    int failedTrials = 0;
//    const int INIT_PERIOD = 256;
//    const int MAX_PERIOD = 4096;
//    int SLEEP_PERIOD = INIT_PERIOD;
//    unsigned partsScattered = 0;
//
//    partsScatteredTable = new bool[numLambdasBackward];
//    std::memset(partsScatteredTable, 0, sizeof(bool) * numLambdasBackward);
//
//    // Check queue to see if partition ready
//    while (partsScattered < numLambdasBackward) {
//        consumerQueueLock.lock();
//        if (rangesToScatter.empty()) {
//            consumerQueueLock.unlock();
//            // sleep with backoff
//            usleep(SLEEP_PERIOD); // sleep a little and give up CPUs
//            failedTrials++;
//            if (failedTrials == 64 && SLEEP_PERIOD < MAX_PERIOD) {
//                failedTrials = 0;
//                SLEEP_PERIOD *= 2;
//            }
//        } else {
//            std::pair<unsigned, unsigned> partitionInfo = rangesToScatter.front();
//            rangesToScatter.pop();
//            // Has this partition already been processed
//            if (partsScatteredTable[partitionInfo.first]) {
//                consumerQueueLock.unlock();
//                continue;
//            }
//            partsScatteredTable[partitionInfo.first] = true;
//            consumerQueueLock.unlock();
//
//            // Partition Info: (partId, rowsPerPartition)
//            unsigned startId = partitionInfo.first * partitionInfo.second;
//            unsigned endId = (partitionInfo.first + 1) * partitionInfo.second;
//            endId = endId > graph.localVtxCnt ? graph.localVtxCnt : endId;
//
//            // Create a series of buckets for batching sendout messages to nodes
//            std::vector<unsigned>* batchedIds = new std::vector<unsigned>[numNodes];
//            for (unsigned lvid = startId; lvid < endId; ++lvid) {
//                for (unsigned nid : graph.backwardGhostMap[lvid]) {
//                    batchedIds[nid].push_back(lvid);
//                }
//            }
//
//            // batch sendouts similar to the sequential version
//            bool batchFlag = true;
//            unsigned BATCH_SIZE = std::max(((batchFlag ? MAX_MSG_SIZE : 4096) - DATA_HEADER_SIZE) /
//                                           (sizeof(unsigned) + sizeof(FeatType) * featDim), 1ul); // at least send one vertex
//            for (unsigned nid = 0; nid < numNodes; ++nid) {
//                if (nid == nodeId) {
//                    continue;
//                }
//
//                unsigned backwardGhostVCnt = batchedIds[nid].size();
//                for (unsigned ib = 0; ib < backwardGhostVCnt; ib += BATCH_SIZE) {
//                    unsigned sendBatchSize = (backwardGhostVCnt - ib) < BATCH_SIZE ? (backwardGhostVCnt - ib) : BATCH_SIZE;
//
//                    backwardVerticesPushOut(nid, sendBatchSize, batchedIds[nid].data() + ib, inputTensor, featDim);
//                    recvCntLock.lock();
//                    recvCnt++;
//                    recvCntLock.unlock();
//                }
//            }
//
//            delete[] batchedIds;
//            failedTrials = 0;
//            SLEEP_PERIOD = INIT_PERIOD;
//            partsScattered++;
//        }
//    }
//
//    // Once all partitions scattered, wait on all acks
//    recvCntLock.lock();
//    if (recvCnt > 0) {
//        recvCntCond.wait();
//    }
//    recvCntLock.unlock();
}

inline void
Engine::backwardVerticesPushOut(unsigned receiver, unsigned totCnt, unsigned *lvids, FeatType *gradTensor, unsigned featDim) {
    zmq::message_t msg(DATA_HEADER_SIZE + (sizeof(unsigned) + sizeof(FeatType) * featDim) * totCnt);
    char *msgPtr = (char *)(msg.data());
    sprintf(msgPtr, NODE_ID_HEADER, receiver);
    msgPtr += NODE_ID_DIGITS;
    *(unsigned*)msgPtr = nodeId;
    msgPtr += sizeof(unsigned);
    *(unsigned*)msgPtr = totCnt;;
    msgPtr += sizeof(unsigned);

    for (unsigned i = 0; i < totCnt; ++i) {
        *(unsigned *)msgPtr = graph.localToGlobalId[lvids[i]];
        msgPtr += sizeof(unsigned);
        FeatType *dataPtr = getVtxFeat(gradTensor, lvids[i], featDim);
        memcpy(msgPtr, dataPtr, sizeof(FeatType) * featDim);
        msgPtr += sizeof(FeatType) * featDim;
    }
    commManager.rawMsgPushOut(msg);
}


/**
 *
 * Major part of the engine's communication logic is done by data threads.
 * These threads loop asynchronously with computation workers.
 *
 */
void
Engine::backwardGhostReceiver(unsigned tid) {
    // backoff sleep strategy to improve CPU utilization
    int failedTrials = 0;
    const int INIT_PERIOD = 256;
    const int MAX_PERIOD = 4096;
    int SLEEP_PERIOD = INIT_PERIOD;
    unsigned sender, topic;
    unsigned vtcsRecvd = 0;
    unsigned featDim = getFeatDim(layer);
    FeatType *msgBuf = (FeatType *)new char[MAX_MSG_SIZE];

    // While loop, looping infinitely to get the next message.
    while (!commHalt) {
        // No message in queue.
        if (!commManager.dataPullIn(&sender, &topic, msgBuf, MAX_MSG_SIZE)) {
            // Computation workers done their work, so communicator goes to death as well.
            if (commHalt) {
                delete[] msgBuf;
                // Better to use return than break for compiler optimization
                return;
            }

            usleep(SLEEP_PERIOD); // sleep a little and give up CPUs
            failedTrials++;
            if (failedTrials == 64 && SLEEP_PERIOD < MAX_PERIOD) {
                failedTrials = 0;
                SLEEP_PERIOD *= 2;
            }
        // Pull in the next message, and process this message.
        } else {
            // A normal ghost value broadcast.
            if (topic < MAX_IDTYPE - 1) {
                // Using MAX_IDTYPE - 1 as the receive signal.
                commManager.dataPushOut(sender, nodeId, MAX_IDTYPE - 1, NULL, 0);
                vtcsRecvd += topic;

                char *bufPtr = (char *)msgBuf;
                unsigned recvGhostVCnt = topic;
                // Update ghost vertices
                for (unsigned i = 0; i < recvGhostVCnt; ++i) {
                    unsigned gvid = *(unsigned *)bufPtr;
                    bufPtr += sizeof(unsigned);
                    FeatType *dataPtr = getVtxFeat(backwardGhostVerticesDataOut, graph.dstGhostVtcs[gvid] - graph.localVtxCnt, featDim);
                    memcpy(dataPtr, bufPtr, sizeof(FeatType) * featDim);
                    bufPtr += sizeof(FeatType) * featDim;
                }

                // A respond to a broadcast, and the topic vertex is in my local vertices. I should update the
                // corresponding recvWaiter's value. If waiters become empty, send a signal in case the workers are
                // waiting on it to be empty at the layer barrier.
            } else { // (topic == MAX_IDTYPE - 1)
                recvCntLock.lock();
                recvCnt--;
                recvCntLock.unlock();
            }
            recvCntLock.lock();
            if (recvCnt == 0 && vtcsRecvd == graph.dstGhostCnt) {
                recvCntCond.signal();
            }
            recvCntLock.unlock();

            SLEEP_PERIOD = INIT_PERIOD;
            failedTrials = 0;
        }
    }
    delete[] msgBuf;
}

/**
 *
 * Calculate batch loss and accuracy based on forward predicts and labels locally.
 */
inline void
Engine::calcAcc(FeatType *predicts, FeatType *labels, unsigned vtcsCnt, unsigned featDim) {
    float acc = 0.0;
    float loss = 0.0;
    for (unsigned i = 0; i < vtcsCnt; i++) {
        FeatType *currLabel = labels + i * featDim;
        FeatType *currPred = predicts + i * featDim;
        acc += currLabel[argmax(currPred, currPred + featDim)];
        loss -= std::log(currPred[argmax(currLabel, currLabel + featDim)]);
    }
    acc /= vtcsCnt;
    loss /= vtcsCnt;
    printLog(getNodeId(), "batch loss %f, batch acc %f", loss, acc);

    accuracy = acc;
}

void Engine::saveTensor(std::string& name, unsigned rows, unsigned cols, FeatType* dptr) {
    auto iter = savedVtxTensors.find(name);
    if (iter != savedVtxTensors.end()) {
        delete[] (iter->second).getData();
        savedVtxTensors.erase(iter);
    }
    savedVtxTensors[name] = Matrix(name.c_str(), rows, cols, dptr);
}

void Engine::saveTensor(const char* name, unsigned rows, unsigned cols, FeatType* dptr) {
    auto iter = savedVtxTensors.find(name);
    if (iter != savedVtxTensors.end()) {
        delete[] (iter->second).getData();
        savedVtxTensors.erase(iter);
    }
    savedVtxTensors[std::string(name)] = Matrix(name, rows, cols, dptr);
}

void Engine::saveTensor(Matrix& mat) {
    auto iter = savedVtxTensors.find(mat.name());
    if (iter != savedVtxTensors.end()) {
        delete[] (iter->second).getData();
        savedVtxTensors.erase(iter);
    }
    savedVtxTensors[mat.name()] = mat;
}

void Engine::saveTensor(const char* name, unsigned layer, unsigned rows, unsigned cols, FeatType* dptr) {
    savedNNTensors[layer][std::string(name)] = Matrix(rows, cols, dptr);
}

void Engine::saveTensor(const char* name, unsigned layer, Matrix& mat) {
    savedNNTensors[layer][std::string(name)] = mat;
}


/**
 *
 * Print engine metrics of processing time.
 *
 */
void
Engine::printEngineMetrics() {
    gtimers.report();
    printLog(nodeId, "<EM>: Using %u forward lambdas and %u bacward lambdas",
             numLambdasForward, numLambdasBackward);
    printLog(nodeId, "<EM>: Initialization takes %.3lf ms", timeInit);
    if (!pipeline) {
        printLog(nodeId, "<EM>: Forward:  Time per stage:");
        for (unsigned i = 0; i < numLayers; ++i) {
            printLog(nodeId, "<EM>    Aggregation   %2u  %.3lf ms", i, vecTimeAggregate[i] / (float)numEpochs);
            printLog(nodeId, "<EM>    ApplyVertex   %2u  %.3lf ms", i, vecTimeApplyVtx[i] / (float)numEpochs);
            printLog(nodeId, "<EM>    Scatter       %2u  %.3lf ms", i, vecTimeScatter[i] / (float)numEpochs);
            printLog(nodeId, "<EM>    ApplyEdge     %2u  %.3lf ms", i, vecTimeApplyEdg[i] / (float)numEpochs);
        }
    }
    printLog(nodeId, "<EM>: Total forward-prop time %.3lf ms", timeForwardProcess / (float)numEpochs);

    printLog(nodeId, "<EM>: Backward: Time per stage:");
    if (!pipeline) {
        for (unsigned i = numLayers; i < 2 * numLayers; i++) {
            printLog(nodeId, "<EM>    Aggregation   %2u  %.3lf ms", i, vecTimeAggregate[i] / (float)numEpochs);
            printLog(nodeId, "<EM>    ApplyVertex   %2u  %.3lf ms", i, vecTimeApplyVtx[i] / (float)numEpochs);
            printLog(nodeId, "<EM>    Scatter       %2u  %.3lf ms", i, vecTimeScatter[i] / (float)numEpochs);
            printLog(nodeId, "<EM>    ApplyEdge     %2u  %.3lf ms", i, vecTimeApplyEdg[i] / (float)numEpochs);
        }
    }
    printLog(nodeId, "<EM>: Total backward-prop time %.3lf ms", timeBackwardProcess / (float)numEpochs);

    double sum = 0.0;
    for (double& d : epochTimes) sum += d;
    printLog(nodeId, "<EM>: Average epoch time %.3lf ms", sum / (float)numEpochs);
    printLog(nodeId, "<EM>: Final accuracy %.3lf", accuracy);

    printLog(nodeId, "Relaunched Lambda Cnt: %u", resComm->getRelaunchCnt());
}


/**
 *
 * Print my graph's metrics.
 *
 */
void
Engine::printGraphMetrics() {
    printLog(nodeId, "<GM>: %u global vertices, %llu global edges, %u local vertices.",
             graph.globalVtxCnt, graph.globalEdgeCnt, graph.localVtxCnt);
}


/**
 *
 * Parse command line arguments.
 *
 */
void
Engine::parseArgs(int argc, char *argv[]) {
    boost::program_options::options_description desc("Allowed options");
    desc.add_options()
    ("help", "Produce help message")

    ("datasetdir", boost::program_options::value<std::string>(), "Path to the dataset")
    ("featuresfile", boost::program_options::value<std::string>(), "Path to the file containing the vertex features")
    ("layerfile", boost::program_options::value<std::string>(), "Layer configuration file")
    ("labelsfile", boost::program_options::value<std::string>(), "Target labels file")
    ("dshmachinesfile", boost::program_options::value<std::string>(), "DSH machines file")
    ("pripfile", boost::program_options::value<std::string>(), "File containing my private ip")
    ("pubipfile", boost::program_options::value<std::string>(), "File containing my public ip")

    ("tmpdir", boost::program_options::value<std::string>(), "Temporary directory")

    ("dataserverport", boost::program_options::value<unsigned>(), "The port exposing to the lambdas")
    ("weightserverport", boost::program_options::value<unsigned>(), "The port of the listener on the lambdas")
    ("wserveripfile", boost::program_options::value<std::string>(), "The file contains the public IP addresses of the weight server")

    // Default is directed graph!
    ("undirected", boost::program_options::value<unsigned>()->default_value(unsigned(0), "0"), "Graph type is undirected or not")

    ("dthreads", boost::program_options::value<unsigned>(), "Number of data threads")
    ("cthreads", boost::program_options::value<unsigned>(), "Number of compute threads")

    ("dataport", boost::program_options::value<unsigned>(), "Port for data communication")
    ("ctrlport", boost::program_options::value<unsigned>(), "Port start for control communication")
    ("nodeport", boost::program_options::value<unsigned>(), "Port for node manager")

    ("numlambdasforward", boost::program_options::value<unsigned>()->default_value(unsigned(1), "5"), "Number of lambdas to request at forward")
    ("numlambdasbackward", boost::program_options::value<unsigned>()->default_value(unsigned(1), "20"), "Number of lambdas to request at backward")
    ("numEpochs", boost::program_options::value<unsigned>(), "Number of epochs to run")
    ("validationFrequency", boost::program_options::value<unsigned>(), "Number of epochs to run before validation")

    ("MODE", boost::program_options::value<unsigned>(), "0: Lambda, 1: GPU, 2: CPU")
    ("pipeline", boost::program_options::value<bool>(), "0: Sequential, 1: Pipelined")
    ("staleness", boost::program_options::value<unsigned>()->default_value(unsigned(UINT_MAX)),
      "Bound on staleness")
    ;

    boost::program_options::variables_map vm;
    boost::program_options::store(boost::program_options::command_line_parser(argc, argv).options(desc).allow_unregistered().run(), vm);
    boost::program_options::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << std::endl;
        exit(-1);
    }

    assert(vm.count("dthreads"));
    dThreads = vm["dthreads"].as<unsigned>();   // Communicator threads.

    assert(vm.count("cthreads"));
    cThreads = vm["cthreads"].as<unsigned>();   // Computation threads.

    assert(vm.count("datasetdir"));
    datasetDir = vm["datasetdir"].as<std::string>();

    assert(vm.count("featuresfile"));
    featuresFile = vm["featuresfile"].as<std::string>();

    assert(vm.count("layerfile"));
    layerConfigFile = vm["layerfile"].as<std::string>();

    assert(vm.count("labelsfile"));
    labelsFile = vm["labelsfile"].as<std::string>();

    assert(vm.count("dshmachinesfile"));
    dshMachinesFile = vm["dshmachinesfile"].as<std::string>();

    assert(vm.count("pripfile"));
    myPrIpFile = vm["pripfile"].as<std::string>();

    assert(vm.count("pubipfile"));
    myPubIpFile = vm["pubipfile"].as<std::string>();

    assert(vm.count("tmpdir"));
    outFile = vm["tmpdir"].as<std::string>() + "/output_";  // Still needs to append the node id, after node manager set up.

    assert(vm.count("dataserverport"));
    dataserverPort = vm["dataserverport"].as<unsigned>();

    assert(vm.count("weightserverport"));
    weightserverPort = vm["weightserverport"].as<unsigned>();

    assert(vm.count("wserveripfile"));
    weightserverIPFile = vm["wserveripfile"].as<std::string>();

    assert(vm.count("undirected"));
    undirected = (vm["undirected"].as<unsigned>() == 0) ? false : true;

    assert(vm.count("dataport"));
    unsigned data_port = vm["dataport"].as<unsigned>();
    commManager.setDataPort(data_port);

    assert(vm.count("ctrlport"));
    unsigned ctrl_port = vm["ctrlport"].as<unsigned>();
    commManager.setControlPortStart(ctrl_port);

    assert(vm.count("nodeport"));
    unsigned node_port = vm["nodeport"].as<unsigned>();
    nodeManager.setNodePort(node_port);

    assert(vm.count("numlambdasforward"));
    numLambdasForward = vm["numlambdasforward"].as<unsigned>();

    assert(vm.count("numlambdasbackward"));
    numLambdasBackward = vm["numlambdasbackward"].as<unsigned>();

    assert(vm.count("numEpochs"));
    numEpochs = vm["numEpochs"].as<unsigned>();

    assert(vm.count("validationFrequency"));
    valFreq = vm["validationFrequency"].as<unsigned>();

    assert(vm.count("MODE"));
    mode = vm["MODE"].as<unsigned>();

    assert(vm.count("pipeline"));
    pipeline = vm["pipeline"].as<bool>();

    assert(vm.count("staleness"));
    staleness = vm["staleness"].as<unsigned>();

    printLog(404, "Parsed configuration: dThreads = %u, cThreads = %u, datasetDir = %s, featuresFile = %s, dshMachinesFile = %s, "
             "myPrIpFile = %s, myPubIpFile = %s, undirected = %s, data port set -> %u, control port set -> %u, node port set -> %u",
             dThreads, cThreads, datasetDir.c_str(), featuresFile.c_str(), dshMachinesFile.c_str(),
             myPrIpFile.c_str(), myPubIpFile.c_str(), undirected ? "true" : "false", data_port, ctrl_port, node_port);
}


/**
 *
 * Read in the layer configuration file.
 *
 */
void
Engine::readLayerConfigFile(std::string &layerConfigFileName) {
    std::ifstream infile(layerConfigFileName.c_str());
    if (!infile.good())
        printLog(nodeId, "Cannot open layer configuration file: %s [Reason: %s]", layerConfigFileName.c_str(), std::strerror(errno));

    assert(infile.good());

    // Loop through each line.
    std::string line;
    while (!infile.eof()) {
        std::getline(infile, line);
        boost::algorithm::trim(line);

        if (line.length() > 0)
            layerConfig.push_back(std::stoul(line));
    }

    assert(layerConfig.size() > 1);
}


/**
 *
 * Read in the initial features file.
 *
 */
void
Engine::readFeaturesFile(std::string &featuresFileName) {
    std::ifstream infile(featuresFileName.c_str());
    if (!infile.good())
        printLog(nodeId, "Cannot open features file: %s [Reason: %s]", featuresFileName.c_str(), std::strerror(errno));

    assert(infile.good());

    FeaturesHeaderType fHeader;
    infile.read((char *) &fHeader, sizeof(FeaturesHeaderType));
    assert(fHeader.numFeatures == layerConfig[0]);

    unsigned gvid = 0;

    unsigned featDim = fHeader.numFeatures;
    std::vector<FeatType> feature_vec;

    feature_vec.resize(featDim);
    while (infile.read(reinterpret_cast<char *> (&feature_vec[0]), sizeof(FeatType) * featDim)) {
        // Set the vertex's initial values, if it is one of my local vertices / ghost vertices.
        if (graph.containsSrcGhostVtx(gvid)) { // Ghost vertex.
            FeatType *actDataPtr = getVtxFeat(forwardGhostInitData, graph.srcGhostVtcs[gvid] - graph.localVtxCnt, featDim);
            memcpy(actDataPtr, feature_vec.data(), featDim * sizeof(FeatType));
        } else if (graph.containsVtx(gvid)) {  // Local vertex.
            FeatType *actDataPtr = getVtxFeat(forwardVerticesInitData, graph.globaltoLocalId[gvid], featDim);
            memcpy(actDataPtr, feature_vec.data(), featDim * sizeof(FeatType));
        }
        ++gvid;
    }
    infile.close();
    assert(gvid == graph.globalVtxCnt);
}


/**
 *
 * Read in the labels file, store the labels in one-hot format.
 *
 */
void
Engine::readLabelsFile(std::string &labelsFileName) {
    std::ifstream infile(labelsFileName.c_str());
    if (!infile.good())
        printLog(nodeId, "Cannot open labels file: %s [Reason: %s]", labelsFileName.c_str(), std::strerror(errno));

    assert(infile.good());

    LabelsHeaderType fHeader;
    infile.read((char *) &fHeader, sizeof(LabelsHeaderType));
    assert(fHeader.labelKinds == layerConfig[numLayers]);

    unsigned gvid = 0;

    unsigned lKinds = fHeader.labelKinds;
    unsigned curr;
    FeatType one_hot_arr[lKinds] = {0};

    while (infile.read(reinterpret_cast<char *> (&curr), sizeof(unsigned))) {
        // Set the vertex's label values, if it is one of my local vertices & is labeled.
        if (graph.containsVtx(gvid)) {
            // Convert into a one-hot array.
            assert(curr < lKinds);
            memset(one_hot_arr, 0, lKinds * sizeof(FeatType));
            one_hot_arr[curr] = 1.0;

            FeatType *labelPtr = localVertexLabelsPtr(graph.globaltoLocalId[gvid]);
            memcpy(labelPtr, one_hot_arr, lKinds * sizeof(FeatType));
        }

        ++gvid;
    }

    infile.close();
    assert(gvid == graph.globalVtxCnt);
}

void
Engine::loadChunks() {
    unsigned vtcsCnt = graph.localVtxCnt;
    for (unsigned cid = 0; cid < numLambdasForward; ++cid) {
        unsigned chunkSize = (vtcsCnt + numLambdasForward - 1) / numLambdasForward;
        unsigned lowBound = cid * chunkSize;
        unsigned upBound = std::min(lowBound + chunkSize, vtcsCnt);

        aggregateQueue.push(Chunk{cid, lowBound, upBound, 0, PROP_TYPE::FORWARD, 1, true});
    }

    // Set the initial bound chunk as epoch 1 layer 0
    minEpoch = 1;
    memset(numFinishedEpoch.data(), 0, sizeof(unsigned) * numFinishedEpoch.size());
}

Engine engine;
