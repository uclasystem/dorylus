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
#include "../GPU-Computation/comp_unit.cuh"
#include "../../common/utils.hpp"
static CuMatrix *NormAdjMatrixIn = NULL;
static CuMatrix *NormAdjMatrixOut = NULL;
static ComputingUnit cu = ComputingUnit::getInstance();
#endif

static int globalEpoch = -1;

// ======== Debug utils ========
typedef std::vector< std::vector<unsigned> > opTimes;
opTimes fwdAggTimes(numLayers);
opTimes fwdInvTimes(numLayers);
opTimes fwdScatrTimes(numLayers);
opTimes bkwdAggTimes(numLayers);
opTimes bkwdInvTimes(numLayers);
opTimes bkwdScatTimes(numLayers);

// Outputs a matrix "signature" (the sum of the elements) without
// creating an actual matrix
void signFile(std::ofstream& outfile, const char* name, FeatType* fptr,
  unsigned start, unsigned end, unsigned c) {
    auto matxSum = [](FeatType* ptr, unsigned start, unsigned end, unsigned c) {
        float sum = 0;
        for (unsigned u = start; u < end; ++u) {
            for (unsigned uj = 0; uj < c; ++uj) {
                sum += ptr[u*c + uj];
            }
        }
        return sum;
    };

    std::stringstream output;
    output << "Matrix: " << name << std::endl;
    output << "Sum: " << matxSum(fptr, start, end, c) << std::endl;

    std::string res = output.str();

    fileMutex.lock();
    outfile.write(res.c_str(), res.size());
    outfile << std::endl << std::flush;
    fileMutex.unlock();
}

// Outputs a full matrix to a file
void outputToFile(std::ofstream& outfile, FeatType* fptr, unsigned start,
  unsigned end, unsigned c) {
    std::string out = "";
    for (uint32_t u = start; u < end; ++u) {
        for (uint32_t uj = 0; uj < c; ++uj) {
            out += std::to_string(fptr[u * c + uj]) + " ";
        }
        out += "\n";
    }
    out += "\n";

    fileMutex.lock();
    outfile.write(out.c_str(), out.size());
    outfile << std::endl << std::flush;
    fileMutex.unlock();
}

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

    // Save intermediate tensors during forward phase for backward computation.
    savedTensors = new std::vector<Matrix>[numLayers];

    // Init it here for collecting data when reading files
    forwardVerticesInitData = new FeatType[getFeatDim(0) * graph.localVtxCnt];
    forwardGhostInitData = new FeatType[getFeatDim(0) * graph.srcGhostCnt];
    // Create labels storage area. Read in labels and store as one-hot format.
    localVerticesLabels = new FeatType[layerConfig[numLayers] * graph.localVtxCnt];

    // Read in initial feature values (input features) & labels.
    readFeaturesFile(featuresFile);
    readLabelsFile(labelsFile);
    saveTensor("LAB", graph.localVtxCnt, layerConfig[numLayers], localVerticesLabels);

    // Initialize synchronization utilities.
    fwdRecvCnt = 0;
    fwdRecvCntLock.init();
    fwdRecvCntCond.init(fwdRecvCntLock);
    bkwdRecvCnt = 0;
    bkwdRecvCntLock.init();
    bkwdRecvCntCond.init(bkwdRecvCntLock);

    consumerQueueLock.init();

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

    // Creating a function pointer so lambda workers can call back to scatter
    // on return
    setupCommInfo();

    resComm = createResourceComm(mode, commInfo);

    timeForwardProcess = 0.0;
    timeInit += getTimer();
    printLog(nodeId, "Engine initialization complete.");
    start_time = getCurrentTime();
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

    fwdRecvCntLock.destroy();
    fwdRecvCntLock.destroy();
    bkwdRecvCntCond.destroy();
    bkwdRecvCntCond.destroy();

    resComm->sendShutdownMessage();

    destroyResourceComm(mode, resComm);

    delete[] forwardVerticesInitData;
    delete[] forwardGhostInitData;

    delete[] localVerticesLabels;
}


/**
*
* Initialize the CommInfo struct for lambdaComm and GPUComm
*
*/
void
Engine::setupCommInfo() {
    commInfo.nodeIp = nodeManager.getNode(nodeId).prip;
    commInfo.nodeId = nodeId;
    commInfo.dataserverPort = dataserverPort;
    commInfo.numLambdasForward = numLambdasForward;
    commInfo.numLambdasBackward = numLambdasBackward;
    commInfo.numNodes = numNodes;
    commInfo.wServersFile = weightserverIPFile;
    commInfo.weightserverPort = weightserverPort;
    commInfo.totalLayers = numLayers;
    commInfo.queuePtr = &rangesToScatter;
    commInfo.savedVtxTensors = &savedVtxTensors;
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
    globalEpoch = epoch;
    // Make sure all nodes start running the forward-prop phase.
    // nodeManager.barrier();
    printLog(nodeId, "Engine starts running FORWARD...");

    timeForwardProcess -= getTimer();

    // Create buffer for first-layer aggregation.
    FeatType *inputTensor = forwardVerticesInitData;
    forwardGhostVerticesDataIn = forwardGhostInitData;

    // For sequential invocation of operations use this block
    if (!pipeline) {
        for (iteration = 0; iteration < numLayers; ++iteration) {
            inputTensor = aggregate(inputTensor, graph.localVtxCnt, getFeatDim(iteration));

            inputTensor = invokeLambda(inputTensor, graph.localVtxCnt,
              getFeatDim(iteration), getFeatDim(iteration + 1));
            if (iteration < numLayers - 1) { // don't need scatter at the last layer.
                inputTensor = scatter(inputTensor,
                  graph.localVtxCnt, getFeatDim(iteration + 1));
                forwardGhostVerticesDataIn = forwardGhostVerticesDataOut;
            }
        }
    }

    // For fused gather and apply, use this block
    // if (pipeline) {
    //     for (iteration = 0; iteration < numLayers; ++iteration) {
    //         inputTensor = fusedGatherApply(inputTensor, graph.localVtxCnt,
    //             getFeatDim(iteration), getFeatDim(iteration + 1));
    //         if (iteration < numLayers - 1) { // don't need scatter at the last layer.
    //             inputTensor = scatter(inputTensor, graph.localVtxCnt, getFeatDim(iteration + 1));
    //             forwardGhostVerticesDataIn = forwardGhostVerticesDataOut;
    //         }
    //     }
    // }

    // For Aggregation -> Apply -> Scatter pipeline use this block
    if (pipeline) {
        for (iteration = 0; iteration < numLayers; ++iteration) {
            inputTensor = fusedGAS(inputTensor, graph.localVtxCnt, getFeatDim(iteration),
              getFeatDim(iteration + 1), iteration < numLayers - 1);
        }
    }

    // TODO: Implement Agg_0 -> Apply_0 -> Scatter_0 -> Agg_1 ... pipeline
    // {
    // }

    timeForwardProcess += getTimer();
    printLog(nodeId, "Engine completes FORWARD at iter %u.", iteration);
    calcAcc(inputTensor, localVerticesLabels, graph.localVtxCnt, getFeatDim(numLayers));

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
    // nodeManager.barrier();
    printLog(nodeId, "Engine starts running BACKWARD...");

    timeBackwardProcess -= getTimer();

    // Create buffer for first-layer aggregation.
    FeatType *gradTensor = initGradTensor;
    // TODO: (YIFAN) this backward flow is correct but weird. I know you have thought for a long time, but try again to refine it.
    iteration = numLayers;

    // Pure sequential
    if (!pipeline) {
        gradTensor = invokeLambdaBackward(gradTensor, graph.localVtxCnt, getFeatDim(numLayers - 1), getFeatDim(numLayers));

        for (iteration = numLayers - 1; iteration > 0; --iteration) {
            gradTensor = scatterBackward(gradTensor, graph.localVtxCnt, getFeatDim(iteration));
            backwardGhostVerticesDataIn = backwardGhostVerticesDataOut;

            gradTensor = aggregateBackward(gradTensor, graph.localVtxCnt, getFeatDim(iteration));

            gradTensor = invokeLambdaBackward(gradTensor, graph.localVtxCnt, getFeatDim(iteration - 1), getFeatDim(iteration));
        }
    }

    // Fused Gather-Apply
    // if (pipeline) {
    //     gradTensor = invokeLambdaBackward(gradTensor, graph.localVtxCnt, getFeatDim(numLayers - 1), getFeatDim(numLayers));
    //     for (iteration = numLayers - 1; iteration > 0; --iteration) {
    //         gradTensor = scatterBackward(gradTensor, graph.localVtxCnt, getFeatDim(iteration));
    //         backwardGhostVerticesDataIn = backwardGhostVerticesDataOut;
    //         gradTensor = fusedGatherApplyBackward(gradTensor, graph.localVtxCnt, getFeatDim(iteration - 1), getFeatDim(iteration));
    //     }
    // }

    // Fused GAS Backward
    if (pipeline) {
        for (iteration = numLayers; iteration > 0; --iteration) {
            gradTensor = fusedGASBackward(gradTensor, graph.localVtxCnt, getFeatDim(iteration - 1), getFeatDim(iteration), iteration < numLayers, iteration > 1);
        }
    }
    delete[] gradTensor;

    timeBackwardProcess += getTimer();
    printLog(nodeId, "Engine completes BACKWARD at iter %u.", iteration);
}


void
Engine::runGCN() {
    auto avg = [&](std::vector<unsigned>& vec) {
        unsigned sum = 0;
        for (auto& u : vec) sum += u;

        return (float)sum / (float)vec.size();
    };

    char tName[8];
    sprintf(tName, "H%d", -1);
    saveTensor(tName, graph.localVtxCnt, getFeatDim(0), forwardVerticesInitData);
    for (unsigned epoch = 0; epoch < numEpochs; ++epoch) {
        globalEpoch = epoch;
        unsigned epStart = timestamp_ms();
        forwardGhostVerticesDataIn = forwardGhostInitData;
        nodeManager.barrier();

        for (iteration = 0; iteration < numLayers; ++iteration) {
            unsigned featDim = getFeatDim(iteration);
            unsigned nextFeatDim = getFeatDim(iteration + 1);

            // AGGREGATE FORWARD
            unsigned aggStart = timestamp_ms();
            FeatType* ahTensor = new FeatType[graph.localVtxCnt * featDim];
            sprintf(tName, "AH%u", iteration);
            saveTensor(tName, graph.localVtxCnt, featDim, ahTensor);

            Matrix& input = savedVtxTensors.find("H" + std::to_string((int(iteration)) - 1))->second;
            FeatType* inputTensor = input.getData();
            AggOPArgs args = {ahTensor, inputTensor, graph.localVtxCnt, featDim};
            auto computeFn = std::bind(&Engine::aggregateCompute, this, std::placeholders::_1,
                    std::placeholders::_2);

            currId = 0;
            computePool->perform(computeFn, &args);
            computePool->sync();
            unsigned aggEnd = timestamp_ms();
            fwdAggTimes[iteration].push_back(aggEnd - aggStart);
            printLog(nodeId, "FWD Aggregate %u took %u ms", iteration, aggEnd - aggStart);

            if (iteration > 0)
                delete[] forwardGhostVerticesDataIn;

            // INVOKE FORWARD
            unsigned invStart = timestamp_ms();
            if (iteration < numLayers - 1) {
                // If middle layer, allocate Z and H tensors
                FeatType* zTensor = new FeatType[graph.localVtxCnt * nextFeatDim];
                FeatType* hTensor = new FeatType[graph.localVtxCnt * nextFeatDim];
                sprintf(tName, "Z%u", iteration);
                saveTensor(tName, graph.localVtxCnt, nextFeatDim, zTensor);
                sprintf(tName, "H%u", iteration);
                saveTensor(tName, graph.localVtxCnt, nextFeatDim, hTensor);
            } else {
                // If final layer, pre-allocate backward tensor for backward
                FeatType* gradTensor = new FeatType[graph.localVtxCnt * featDim];
                sprintf(tName, "GRAD%u", iteration);
                saveTensor(tName, graph.localVtxCnt, featDim, gradTensor);
                resComm->sendInfoMsg(iteration);
            }

            resComm->reset(iteration);
            for (unsigned u = 0; u < numLambdasForward; ++u) {
                resComm->requestInvoke(iteration, u, PROP_TYPE::FORWARD, iteration == numLayers - 1);
            }

            resComm->waitLambda(iteration, PROP_TYPE::FORWARD, iteration == numLayers - 1);
            unsigned invEnd = timestamp_ms();
            fwdInvTimes[iteration].push_back(invEnd - invStart);
            printLog(nodeId, "FWD Invoke %u took %u ms", iteration, invEnd - invStart);

            // SCATTER FORWARD
            if (iteration < numLayers - 1) {
                unsigned scatStart = timestamp_ms();
                FeatType* hTensor = (savedVtxTensors.find("H" + std::to_string(iteration))->second).getData();
                scatter(hTensor, graph.localVtxCnt, nextFeatDim);
                forwardGhostVerticesDataIn = forwardGhostVerticesDataOut;
                unsigned scatEnd = timestamp_ms();
                fwdScatrTimes[iteration].push_back(scatEnd - scatStart);
                printLog(nodeId, "FWD Scatter %u took %u ms", iteration, scatEnd - scatStart);
            }
        }

        for (iteration = numLayers - 1; iteration > 0; --iteration) {
            unsigned featDim = getFeatDim(iteration);
            unsigned prevFeatDim = getFeatDim(iteration - 1);

            // SCATTER BACKWARD
            unsigned scatStart = timestamp_ms();
            FeatType* outputGradTensor = ((savedVtxTensors.find("GRAD" + std::to_string(iteration))->second)).getData();
            scatterBackward(outputGradTensor, graph.localVtxCnt, featDim);
            backwardGhostVerticesDataIn = backwardGhostVerticesDataOut;
            unsigned scatEnd = timestamp_ms();
            bkwdScatTimes[iteration].push_back(scatEnd - scatStart);
            printLog(nodeId, "BKWD Scatter %u took %u ms", iteration, scatEnd - scatStart);

            // AGGREGATE BACKWARD
            unsigned aggStart = timestamp_ms();
            FeatType* bahTensor = new FeatType[graph.localVtxCnt * featDim];
            sprintf(tName, "BAH%u", iteration);
            saveTensor(tName, graph.localVtxCnt, featDim, bahTensor);
            currId = 0;
            AggOPArgs args = {bahTensor, outputGradTensor, graph.localVtxCnt, featDim};
            auto computeFn = std::bind(&Engine::aggregateBPCompute, this, std::placeholders::_1, std::placeholders::_2);
            computePool->perform(computeFn, &args);
            computePool->sync();
            unsigned aggEnd = timestamp_ms();
            bkwdAggTimes[iteration].push_back(aggEnd - aggStart);
            printLog(nodeId, "BKWD Aggregate %u took %u ms", iteration, aggEnd - aggStart);

            // INVOKE BACKWARD
            unsigned invStart = timestamp_ms();
            FeatType* gradTensor = new FeatType[graph.localVtxCnt * prevFeatDim];
            sprintf(tName, "GRAD%u", iteration-1);
            saveTensor(tName, graph.localVtxCnt, featDim, gradTensor);

            resComm->sendInfoMsg(iteration-1);
            resComm->reset(iteration-1);
            for (unsigned u = 0; u < numLambdasForward; ++u) {
                resComm->requestInvoke(iteration-1, u, PROP_TYPE::BACKWARD, iteration == numLayers - 1);
            }

            resComm->waitLambda(iteration-1, PROP_TYPE::BACKWARD, iteration == numLayers - 1);
            unsigned invEnd = timestamp_ms();
            bkwdInvTimes[iteration].push_back(invEnd - invStart);
            printLog(nodeId, "BKWD Invoke %u took %u ms", iteration-1, invEnd - invStart);
        }
        unsigned epEnd = timestamp_ms();
        unsigned epTime = epEnd - epStart;

        printLog(nodeId, "Finished epoch %u. Epoch Time: %u ms", epoch, epTime);
        epochMs.push_back(epTime);
    }

    if (nodeId == 0) {
        std::stringstream output;
        for (unsigned u = 0; u < numLayers; ++u) {
            if (!fwdAggTimes[u].empty())
                output << "<EM> Average fwd agg time for layer " << u
                  << ": " << avg(fwdAggTimes[u]) << " ms" << std::endl;
            if (!fwdInvTimes[u].empty())
                output << "<EM> Average fwd invoke time for layer " << u
                  << ": " << avg(fwdInvTimes[u]) << " ms" << std::endl;
            if (!fwdScatrTimes[u].empty())
                output << "<EM> Average fwd scatter time for layer " << u
                  << ": " << avg(fwdScatrTimes[u]) << " ms" << std::endl;
            if (!bkwdScatTimes[u].empty())
                output << "<EM> Average bkwd agg time for layer " << u
                  << ": " << avg(bkwdScatTimes[u]) << " ms" << std::endl;
            if (!bkwdAggTimes[u].empty())
                output << "<EM> Average bkwd scatter time for layer " << u
                  << ": " << avg(bkwdAggTimes[u]) << " ms" << std::endl;
            if (!bkwdInvTimes[u].empty())
                output << "<EM> Average bkwd invoke for layer " << u
                  << ": " << avg(bkwdInvTimes[u]) << " ms" << std::endl;
        }

        output << "<EM> Average epoch time " << avg(epochMs) << " ms" << std::endl;

        printLog(nodeId, output.str().c_str());
    }

    return;
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
    //               << "L: " << vecTimeLambda[i] << std::endl
    //               << "G: " << vecTimeSendout[i] << std::endl;
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
            sprintf(outBuf, "<EM>    Lambda        %2u  %.3lf ms", i, vecTimeLambda[i] / (float)numEpochs);
            outStream << outBuf << std::endl;
            sprintf(outBuf, "<EM>    Ghost update  %2u  %.3lf ms", i, vecTimeSendout[i] / (float)numEpochs);
            outStream << outBuf << std::endl;
        }
    }
    sprintf(outBuf, "<EM>: Total forward-prop time %.3lf ms", timeForwardProcess / (float)numEpochs);
    outStream << outBuf << std::endl;

    if (!pipeline) {
        sprintf(outBuf, "<EM>: Backward: Time per stage:");
        outStream << outBuf << std::endl;
        for (unsigned i = numEpochs; i < 2 * numLayers; i++) {
            sprintf(outBuf, "<EM>    Aggregation   %2u  %.3lf ms", i, vecTimeAggregate[i] / (float)numEpochs);
            outStream << outBuf << std::endl;
            sprintf(outBuf, "<EM>    Lambda        %2u  %.3lf ms", i, vecTimeLambda[i] / (float)numEpochs);
            outStream << outBuf << std::endl;
            sprintf(outBuf, "<EM>    Ghost update  %2u  %.3lf ms", i, vecTimeSendout[i] / (float)numEpochs);
            outStream << outBuf << std::endl;
        }
    }
    sprintf(outBuf, "<EM>: Backward-prop takes %.3lf ms", timeBackwardProcess);
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
        assert(vecTimeLambda.size() == 2 * numLayers);
        assert(vecTimeSendout.size() == 2 * numLayers);
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

    if (iteration > 0) {
        delete[] forwardGhostVerticesDataIn;
        delete[] vtcsTensor;
    }


    if (vecTimeAggregate.size() < numLayers) {
        vecTimeAggregate.push_back(getTimer() - sttTimer);
    } else {
        vecTimeAggregate[iteration] += getTimer() - sttTimer;
    }

    return outputTensor;
}
#else
FeatType *Engine::aggregate(FeatType *vtcsTensor, unsigned vtcsCnt, unsigned featDim) {
    double sttTimer = getTimer();
    // AH
    FeatType *outputTensor = new FeatType[vtcsCnt * featDim];
    currId = 0;

    AggOPArgs args = {outputTensor, vtcsTensor, vtcsCnt, featDim};
    auto computeFn = std::bind(&Engine::aggregateCompute, this, std::placeholders::_1, std::placeholders::_2);

    computePool->perform(computeFn, &args);
    computePool->sync();

    if (iteration > 0) {
        delete[] forwardGhostVerticesDataIn;
        delete[] vtcsTensor;
    }


    if (vecTimeAggregate.size() < numLayers) {
        vecTimeAggregate.push_back(getTimer() - sttTimer);
    } else {
        vecTimeAggregate[iteration] += getTimer() - sttTimer;
    }
    Matrix m(vtcsCnt, featDim, outputTensor);
    return outputTensor;
}
#endif // _GPU_ENABLED_

FeatType*
Engine::invokeLambda(FeatType *vtcsTensor, unsigned vtcsCnt, unsigned inFeatDim, unsigned outFeatDim) {
    double sttTimer = getTimer();
    assert(vtcsCnt == graph.localVtxCnt);

    FeatType *outputTensor = new FeatType[vtcsCnt * outFeatDim];
    FeatType *zTensor = new FeatType[vtcsCnt * outFeatDim];

    bool saveInput = true;
    if (saveInput) {
        savedTensors[iteration].push_back(Matrix(vtcsCnt, inFeatDim, vtcsTensor));
    }
    savedTensors[iteration].push_back(Matrix(vtcsCnt, outFeatDim, zTensor));

    // Start a new lambda communication context.
    resComm->newContextForward(iteration, vtcsTensor, zTensor, outputTensor, vtcsCnt, inFeatDim, outFeatDim);

    if (mode == LAMBDA) {
        double invTimer = getTimer();
        unsigned availLambdaId = 0;
        while (availLambdaId < numLambdasForward) {
            resComm->invokeLambdaForward(iteration, availLambdaId, iteration == numLayers - 1);
            availLambdaId++;
        }
        if (vecTimeLambdaInvoke.size() < numLayers) {
            vecTimeLambdaInvoke.push_back(getTimer() - invTimer);
        } else {
            vecTimeLambdaInvoke[iteration] += getTimer() - invTimer;
        }
        double waitTimer = getTimer();
        resComm->waitLambdaForward(iteration, iteration == numLayers - 1);
        if (vecTimeLambdaWait.size() < numLayers) {
            vecTimeLambdaWait.push_back(getTimer() - waitTimer);
        } else {
            vecTimeLambdaWait[iteration] += getTimer() - waitTimer;
        }
    }
    // if in GPU mode we launch gpu computation here and wait the results
    else
        resComm->requestForward(iteration, iteration == numLayers - 1);

    bool saveOutput = true;
    if (saveOutput) {
        FeatType *outTensorCpy = new FeatType [vtcsCnt * outFeatDim];
        memcpy(outTensorCpy, outputTensor, vtcsCnt * outFeatDim * sizeof(FeatType));
        savedTensors[iteration].push_back(Matrix(vtcsCnt, outFeatDim, outTensorCpy));
    }

    if (saveInput) {
        vtcsTensor = NULL;
    } else {
        delete[] vtcsTensor;
    }

    if (vecTimeLambda.size() < numLayers) {
        vecTimeLambda.push_back(getTimer() - sttTimer);
    } else {
        vecTimeLambda[iteration] += getTimer() - sttTimer;
    }

    return outputTensor;
}

FeatType*
Engine::fusedGatherApply(FeatType *vtcsTensor, unsigned vtcsCnt, unsigned inFeatDim, unsigned outFeatDim) {
    double sttTimer = getTimer();
    // Prepare for gather phase
    // AH
    FeatType *gatheredTensor = new FeatType[vtcsCnt * inFeatDim];
    currId = 0;
    AggOPArgs args = {gatheredTensor, vtcsTensor, vtcsCnt, inFeatDim};
    auto computeFn = std::bind(&Engine::aggregateCompute, this, std::placeholders::_1, std::placeholders::_2);

    // Start gathering
    computePool->perform(computeFn, &args);

    // Prepare for applyVertex phase
    // H, Z
    FeatType *outputTensor = new FeatType[vtcsCnt * outFeatDim];
    FeatType *zTensor = new FeatType[vtcsCnt * outFeatDim];
    bool saveInput = true;
    if (saveInput) {
        savedTensors[iteration].push_back(Matrix(vtcsCnt, inFeatDim, gatheredTensor));
    }
    savedTensors[iteration].push_back(Matrix(vtcsCnt, outFeatDim, zTensor));
    resComm->newContextForward(iteration, gatheredTensor, zTensor, outputTensor, vtcsCnt, inFeatDim, outFeatDim);

    // Start applyVertex phase
    unsigned currLambdaId = 0;
    if (mode == LAMBDA) {
        const unsigned lambdaChunkSize = (vtcsCnt + numLambdasForward - 1) / numLambdasForward;
        unsigned availChunkSize = lambdaChunkSize;
        while (currId < vtcsCnt) {
            unsigned lvid = currId;
            while (lvid > availChunkSize) {
                resComm->invokeLambdaForward(iteration, currLambdaId, iteration == numLayers - 1);
                ++currLambdaId;
                availChunkSize += lambdaChunkSize;
            }
            usleep(5000); // wait 5ms and then check again
        }
    }
    computePool->sync();
    if (mode != LAMBDA) {
        resComm->requestForward(iteration, iteration == numLayers - 1);
    } else {
        while (currLambdaId < numLambdasForward) {
            resComm->invokeLambdaForward(iteration, currLambdaId, iteration == numLayers - 1);
            ++currLambdaId;
        }
        resComm->waitLambdaForward(iteration, iteration == numLayers - 1);
    }

    // Post-processing for applyVertex phase & clean up
    bool saveOutput = true;
    if (saveOutput) {
        FeatType *outTensorCpy = new FeatType[vtcsCnt * outFeatDim];
        memcpy(outTensorCpy, outputTensor, vtcsCnt * outFeatDim * sizeof(FeatType));
        savedTensors[iteration].push_back(Matrix(vtcsCnt, outFeatDim, outTensorCpy));
    }
    if (saveInput) {
        gatheredTensor = NULL;
    } else {
        delete[] gatheredTensor;
    }

    // Clean up the gather phase
    if (iteration > 0) {
        delete[] forwardGhostVerticesDataIn;
        delete[] vtcsTensor;
    }

    if (vecTimeAggregate.size() < numLayers) {
        vecTimeAggregate.push_back(getTimer() - sttTimer);
    } else {
        vecTimeAggregate[iteration] += getTimer() - sttTimer;
    }

    return outputTensor;
}

FeatType *
Engine::scatter(FeatType *vtcsTensor, unsigned vtcsCnt, unsigned featDim) {
    double sttTimer = getTimer();

    // Start data communicators.
    commHalt = false;
    forwardGhostVerticesDataOut = new FeatType[graph.srcGhostCnt * featDim];
    auto fgc_fp = std::bind(&Engine::forwardGhostReceiver, this, std::placeholders::_1);
    dataPool->perform(fgc_fp);

    sendForwardGhostUpdates(vtcsTensor, featDim);

    // TODO: (YIFAN) we can optimize this to extend comm protocol. Mark the last packet sent so this node knows when to exit ghostCommunicator.
    nodeManager.barrier();

    commHalt = true;
    // Join all data communicators.
    dataPool->sync();

    if (vecTimeSendout.size() < numLayers) {
        vecTimeSendout.push_back(getTimer() - sttTimer);
    } else {
        vecTimeSendout[iteration] += getTimer() - sttTimer;
    }

    return vtcsTensor;
}

FeatType*
Engine::fusedGAS(FeatType* vtcsTensor, unsigned vtcsCnt, unsigned inFeatDim,
  unsigned outFeatDim, bool scatter) {
    double sttTimer = getTimer();
    // Check just to make sure partition ranges are empty
    // before starting
    consumerQueueLock.lock();
    while (!rangesToScatter.empty()) rangesToScatter.pop();
    consumerQueueLock.unlock();

    // Start data receivers
    commHalt = false;
    // Forward declaration to ensure pointer remains in scope
    FeatType *outputTensor = new FeatType[vtcsCnt * outFeatDim];
    FeatType *zTensor = new FeatType[vtcsCnt * outFeatDim];
    auto fgr_fp = std::bind(&Engine::forwardGhostReceiver, this, std::placeholders::_1);
    auto fgu_fp = std::bind(&Engine::pipelineForwardGhostUpdates, this,
                    std::placeholders::_1, std::placeholders::_2);
    std::thread scatterThread;
    if (scatter) {
        forwardGhostVerticesDataOut = new FeatType[graph.srcGhostCnt * outFeatDim];
        dataPool->perform(fgr_fp);
        scatterThread = std::thread(fgu_fp, outputTensor, outFeatDim);
    }

    // Prepare for gather phase
    FeatType *gatheredTensor = new FeatType[vtcsCnt * inFeatDim];
    currId = 0;
    AggOPArgs args = {gatheredTensor, vtcsTensor, vtcsCnt, inFeatDim};
    auto computeFn = std::bind(&Engine::aggregateCompute, this, std::placeholders::_1, std::placeholders::_2);

    // Start gathering
    computePool->perform(computeFn, &args);

    // Prepare for applyVertex phase
    bool saveInput = true;
    if (saveInput) {
        savedTensors[iteration].push_back(Matrix(vtcsCnt, inFeatDim, gatheredTensor));
    }
    savedTensors[iteration].push_back(Matrix(vtcsCnt, outFeatDim, zTensor));
    resComm->newContextForward(iteration, gatheredTensor, zTensor, outputTensor, vtcsCnt, inFeatDim, outFeatDim, scatter);

    // Start applyVertex phase
    unsigned currLambdaId = 0;
    if (mode == LAMBDA) {
        const unsigned lambdaChunkSize = (vtcsCnt + numLambdasForward - 1) / numLambdasForward;
        unsigned availChunkSize = lambdaChunkSize;
        while (currId < vtcsCnt) {
            unsigned lvid = currId;
            while (lvid > availChunkSize) {
                resComm->invokeLambdaForward(iteration, currLambdaId, iteration == numLayers - 1);
                ++currLambdaId;
                availChunkSize += lambdaChunkSize;
            }
            usleep(5000); // wait 5ms and then check again
        }
    }
    computePool->sync();
    if (mode != LAMBDA) {
        resComm->requestForward(iteration, iteration == numLayers - 1);
    } else {
        while (currLambdaId < numLambdasForward) {
            resComm->invokeLambdaForward(iteration, currLambdaId, iteration == numLayers - 1);
            ++currLambdaId;
        }
        resComm->waitLambdaForward(iteration, iteration == numLayers - 1);
    }

    // Wait for all remote schedulings sent by me to be handled.
    if (scatter) {
        scatterThread.join();
    }
    commHalt = true;
    dataPool->sync();

    // Post-processing for applyVertex phase & clean up
    bool saveOutput = true;
    if (saveOutput) {
        FeatType *outTensorCpy = new FeatType[vtcsCnt * outFeatDim];
        memcpy(outTensorCpy, outputTensor, vtcsCnt * outFeatDim * sizeof(FeatType));
        savedTensors[iteration].push_back(Matrix(vtcsCnt, outFeatDim, outTensorCpy));
    }
    if (saveInput) {
        gatheredTensor = NULL;
    } else {
        delete[] gatheredTensor;
    }

    // Clean up the gather phase
    if (iteration > 0) {
        delete[] forwardGhostVerticesDataIn;
        delete[] vtcsTensor;
    }

    if (vecTimeAggregate.size() < numLayers) {
        vecTimeAggregate.push_back(getTimer() - sttTimer);
    } else {
        vecTimeAggregate[iteration] += getTimer() - sttTimer;
    }

    // Set the scattered output as the input for next aggregation phase
    forwardGhostVerticesDataIn = forwardGhostVerticesDataOut;

    return outputTensor;
}


/////////////////////////////////////////////////////////
// Below are private forward functions for the engine. //
/////////////////////////////////////////////////////////
void Engine::aggregateCompute(unsigned tid, void *args) {
    FeatType *outputTensor = ((AggOPArgs *) args)->outputTensor;
    FeatType *vtcsTensor = ((AggOPArgs *) args)->inputTensor;
    const unsigned vtcsCnt = ((AggOPArgs *) args)->vtcsCnt;
    const unsigned featDim = ((AggOPArgs *) args)->featDim;

    unsigned lvid = 0;
    while (currId < vtcsCnt) {
        lvid = __sync_fetch_and_add(&currId, 1);
        if (lvid < vtcsCnt) {
            forwardAggregateFromNeighbors(lvid, outputTensor, vtcsTensor, featDim);
        }
    }
}


/**
 *
 * Aggregate featDim feature values starting from offset from all neighbors (including self). Then write the results to the
 * data buffer area for serialization. The results are to be used for being sent to lambda threads.
 *
 */
inline void
Engine::forwardAggregateFromNeighbors(unsigned lvid, FeatType *outputTensor, FeatType *inputTensor, unsigned featDim) {
    // Read out data of the current iteration of given vertex.
    FeatType *currDataDst = getVtxFeat(outputTensor, lvid, featDim);
    FeatType *currDataPtr = getVtxFeat(inputTensor, lvid, featDim);
    memcpy(currDataDst, currDataPtr, featDim * sizeof(FeatType));

    // Apply normalization factor on the current data.
    {
        const EdgeType normFactor = graph.vtxDataVec[lvid];
        for (unsigned i = 0; i < featDim; ++i) {
            currDataDst[i] *= normFactor;
        }
    }

    // Aggregate from incoming neighbors.
    for (unsigned eid = graph.forwardAdj.columnPtrs[lvid]; eid < graph.forwardAdj.columnPtrs[lvid + 1]; ++eid) {
        FeatType *otherDataPtr;
        EdgeType normFactor = graph.forwardAdj.values[eid];
        unsigned srcVId = graph.forwardAdj.rowIdxs[eid];
        if (srcVId < graph.localVtxCnt) { // local vertex.
            otherDataPtr = getVtxFeat(inputTensor, srcVId, featDim);
        } else {                          // ghost vertex.
            otherDataPtr = getVtxFeat(forwardGhostVerticesDataIn, srcVId - graph.localVtxCnt, featDim);
        }
        for (unsigned j = 0; j < featDim; ++j) {
            currDataDst[j] += otherDataPtr[j] * normFactor;
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
            fwdRecvCntLock.lock();
            fwdRecvCnt++;
            fwdRecvCntLock.unlock();
        }
    }
    // Wait for all remote schedulings sent by me to be handled.
    fwdRecvCntLock.lock();
    if (fwdRecvCnt > 0) {
        fwdRecvCntCond.wait();
    }
    fwdRecvCntLock.unlock();
}

// inputTensor = activation output tensor
inline void
Engine::pipelineForwardGhostUpdates(FeatType* inputTensor, unsigned featDim) {
    int failedTrials = 0;
    const int INIT_PERIOD = 256;
    const int MAX_PERIOD = 4096;
    int SLEEP_PERIOD = INIT_PERIOD;
    unsigned partsScattered = 0;

    partsScatteredTable = new bool[numLambdasForward];
    std::memset(partsScatteredTable, 0, sizeof(bool) * numLambdasForward);

    // Check queue to see if partition ready
    while (partsScattered < numLambdasForward) {
        consumerQueueLock.lock();
        if (rangesToScatter.empty()) {
            consumerQueueLock.unlock();
            // sleep with backoff
            usleep(SLEEP_PERIOD); // sleep a little and give up CPUs
            failedTrials++;
            if (failedTrials == 64 && SLEEP_PERIOD < MAX_PERIOD) {
                failedTrials = 0;
                SLEEP_PERIOD *= 2;
            }
        } else {
            std::pair<unsigned, unsigned> partitionInfo = rangesToScatter.front();
            rangesToScatter.pop();
            // Has this partition already been processed
            if (partsScatteredTable[partitionInfo.first]) {
                consumerQueueLock.unlock();
                continue;
            }
            partsScatteredTable[partitionInfo.first] = true;
            consumerQueueLock.unlock();

            // Partition Info: (partId, rowsPerPartition)
            unsigned startId = partitionInfo.first * partitionInfo.second;
            unsigned endId = (partitionInfo.first + 1) * partitionInfo.second;
            endId = endId > graph.localVtxCnt ? graph.localVtxCnt : endId;

            // Create a series of buckets for batching sendout messages to nodes
            std::vector<unsigned>* batchedIds = new std::vector<unsigned>[numNodes];
            for (unsigned lvid = startId; lvid < endId; ++lvid) {
                for (unsigned nid : graph.forwardGhostMap[lvid]) {
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

                unsigned forwardGhostVCnt = batchedIds[nid].size();
                for (unsigned ib = 0; ib < forwardGhostVCnt; ib += BATCH_SIZE) {
                    unsigned sendBatchSize = (forwardGhostVCnt - ib) < BATCH_SIZE ? (forwardGhostVCnt - ib) : BATCH_SIZE;

                    forwardVerticesPushOut(nid, sendBatchSize, batchedIds[nid].data() + ib, inputTensor, featDim);
                    fwdRecvCntLock.lock();
                    fwdRecvCnt++;
                    fwdRecvCntLock.unlock();
                }
            }

            delete[] batchedIds;
            failedTrials = 0;
            partsScattered++;
        }
    }

    // Once all partitions scattered, wait on all acks
    fwdRecvCntLock.lock();
    if (fwdRecvCnt > 0) {
        fwdRecvCntCond.wait();
    }
    fwdRecvCntLock.unlock();
}

inline void
Engine::forwardVerticesPushOut(unsigned receiver, unsigned totCnt, unsigned *lvids, FeatType *inputTensor, unsigned featDim) {
    zmq::message_t msg(DATA_HEADER_SIZE + (sizeof(unsigned) + sizeof(FeatType) * featDim) * totCnt);
    char *msgPtr = (char *)(msg.data());
    sprintf(msgPtr, NODE_ID_HEADER, receiver);
    msgPtr += NODE_ID_DIGITS;
    *(unsigned *)msgPtr = nodeId; // set sender
    msgPtr += sizeof(unsigned);
    *(unsigned *)msgPtr = totCnt;
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
Engine::forwardGhostReceiver(unsigned tid) {
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
                unsigned featDim = getFeatDim(iteration + 1);
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
                // waiting on it to be empty at the iteration barrier.
            } else { // (topic == MAX_IDTYPE - 1)
                fwdRecvCntLock.lock();
                fwdRecvCnt--;
                fwdRecvCntLock.unlock();
            }
            fwdRecvCntLock.lock();
            if (fwdRecvCnt == 0 && vtcsRecvd == graph.srcGhostCnt) {
                fwdRecvCntCond.signal();
            }
            fwdRecvCntLock.unlock();

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
    vecTimeAggregate[numLayers + iteration] += getTimer() - sttTimer;

    return outputTensor;
}
#else
FeatType*
Engine::aggregateBackward(FeatType *gradTensor, unsigned vtcsCnt, unsigned featDim) {
    double sttTimer = getTimer();

    FeatType *outputTensor = new FeatType[vtcsCnt * featDim];
    currId = 0;

    AggOPArgs args = {outputTensor, gradTensor, vtcsCnt, featDim};
    auto computeFn = std::bind(&Engine::aggregateBPCompute, this, std::placeholders::_1, std::placeholders::_2);
    computePool->perform(computeFn, &args);
    computePool->sync();

    delete[] gradTensor;
    delete[] backwardGhostVerticesDataIn;

    if (vecTimeAggregate.size() < 2 * numLayers) {
        for (unsigned i = vecTimeAggregate.size(); i < 2 * numLayers; i++) {
            vecTimeAggregate.push_back(0.0);
        }
    }
    vecTimeAggregate[numLayers + iteration] += getTimer() - sttTimer;

    return outputTensor;
}
#endif



FeatType*
Engine::invokeLambdaBackward(FeatType *gradTensor, unsigned vtcsCnt, unsigned inFeatDim, unsigned outFeatDim) {
    double sttTimer = getTimer();

    assert(vtcsCnt == graph.localVtxCnt);

    FeatType *outputTensor = new FeatType [vtcsCnt * inFeatDim];

    if (vecTimeLambdaInvoke.size() < 2 * numLayers) {
        for (unsigned i = vecTimeLambdaInvoke.size(); i < 2 * numLayers; ++i) {
            vecTimeLambdaInvoke.push_back(0.0);
            vecTimeLambdaWait.push_back(0.0);
        }
    }

    resComm->newContextBackward(iteration - 1, gradTensor, outputTensor, savedTensors, localVerticesLabels, vtcsCnt, inFeatDim, outFeatDim, getFeatDim(numLayers));
    resComm->requestBackward(iteration - 1, iteration - 1 == numLayers - 1);

    delete[] gradTensor;
    for (auto &sTensor : savedTensors[iteration - 1]) {
        delete[] sTensor.getData();
    }
    savedTensors[iteration - 1].clear();

    if (vecTimeLambda.size() < 2 * numLayers) {
        for (unsigned i = vecTimeLambda.size(); i < 2 * numLayers; i++) {
            vecTimeLambda.push_back(0.0);
        }
    }
    vecTimeLambda[numLayers + iteration - 1] += getTimer() - sttTimer;
    return outputTensor;
}

FeatType*
Engine::fusedGatherApplyBackward(FeatType *gradTensor, unsigned vtcsCnt, unsigned inFeatDim, unsigned outFeatDim) {
    double sttTimer = getTimer();

    // Prepare for gather phase
    FeatType *gatheredTensor = new FeatType[vtcsCnt * outFeatDim];
    FeatType *outputTensor = new FeatType[vtcsCnt * inFeatDim];
    currId = 0;

    // Start gathering
    AggOPArgs args = {gatheredTensor, gradTensor, vtcsCnt, outFeatDim};
    auto computeFn = std::bind(&Engine::aggregateBPCompute, this, std::placeholders::_1, std::placeholders::_2);
    computePool->perform(computeFn, &args);

    // Prepare for applyVertex phase
    resComm->newContextBackward(iteration - 1, gatheredTensor, outputTensor, savedTensors, localVerticesLabels, vtcsCnt, inFeatDim, outFeatDim, getFeatDim(numLayers));

    // Start applyVertex phase
    unsigned currLambdaId = 0;
    if (mode == LAMBDA) {
        const unsigned lambdaChunkSize = (vtcsCnt + numLambdasForward - 1) / numLambdasBackward;
        unsigned availChunkSize = lambdaChunkSize;
        while (currId < vtcsCnt) {
            unsigned lvid = currId;
            if (lvid > availChunkSize) {
                resComm->invokeLambdaBackward(iteration - 1, currLambdaId, iteration - 1 == numLayers - 1);
                availChunkSize += lambdaChunkSize;
                ++currLambdaId;
            }
            usleep(2000); // wait for 2ms and check again
        }
    }
    computePool->sync();
    if (mode != LAMBDA) {
        resComm->requestBackward(iteration - 1, iteration - 1 == numLayers - 1);
    } else {
        while (currLambdaId < numLambdasBackward) {
            resComm->invokeLambdaBackward(iteration - 1, currLambdaId, iteration - 1 == numLayers - 1);
            ++currLambdaId;
        }
        resComm->waitLambdaBackward(iteration - 1, iteration - 1 == numLayers - 1);
    }

    // Clean up applyVertex phase
    delete[] gatheredTensor;
    for (auto &sTensor : savedTensors[iteration - 1]) {
        delete[] sTensor.getData();
    }
    savedTensors[iteration - 1].clear();

    // Clean up gather phase
    delete[] gradTensor;
    delete[] backwardGhostVerticesDataIn;

    if (vecTimeAggregate.size() < 2 * numLayers) {
        for (unsigned i = vecTimeAggregate.size(); i < 2 * numLayers; i++) {
            vecTimeAggregate.push_back(0.0);
        }
    }
    vecTimeAggregate[numLayers + iteration] += getTimer() - sttTimer;

    return outputTensor;
}

FeatType*
Engine::scatterBackward(FeatType *gradTensor, unsigned vtcsCnt, unsigned featDim) {
    double sttTimer = getTimer();

    // Start data communicators.
    commHalt = false;
    bkwdRecvCnt = 0;
    backwardGhostVerticesDataOut = new FeatType[graph.dstGhostCnt * featDim];
    auto bgr_fp = std::bind(&Engine::backwardGhostReceiver, this,
        std::placeholders::_1, std::placeholders::_2);
    dataPool->perform(bgr_fp, (void*) &featDim);

    sendBackwardGhostGradients(gradTensor, featDim);

    //## Global Iteration barrier. ##//
    // TODO: (YIFAN) we can optimize this to extend comm protocal. Mark the last packet sent so this node knows when to exit ghostCommunicator.
    nodeManager.barrier();
    commHalt = true;
    // Join all data communicators.
    dataPool->sync();

    if (vecTimeSendout.size() < 2 * numLayers) {
        for (unsigned i = vecTimeSendout.size(); i < 2 * numLayers; i++) {
            vecTimeSendout.push_back(0.0);
        }
    }
    vecTimeSendout[numLayers + iteration] += getTimer() - sttTimer;
    return gradTensor;
}

FeatType*
Engine::fusedGASBackward(FeatType *gradTensor, unsigned vtcsCnt, unsigned inFeatDim, unsigned outFeatDim, bool aggregate, bool scatter) {
    double sttTimer = getTimer();

    consumerQueueLock.lock();
    while (!rangesToScatter.empty()) rangesToScatter.pop();
    consumerQueueLock.unlock();

    // Case 1 - First phase, no aggregate needed
    FeatType* outputTensor = nullptr;
    if (!aggregate && scatter) {
        outputTensor = applyScatterPhase(gradTensor, vtcsCnt, inFeatDim, outFeatDim, scatter);
    }
    // Case 2 - Full phase including gather, apply, and scatter
    else if (aggregate && scatter) {
        outputTensor = aggregateApplyScatterPhase(gradTensor, vtcsCnt, inFeatDim, outFeatDim, scatter);
    }
    // Case 3 - Final phase, no scatter needed
    else if (aggregate && !scatter) {
        outputTensor = aggregateApplyPhase(gradTensor, vtcsCnt, inFeatDim, outFeatDim, scatter);
    }
    else {
        printLog(nodeId, "\033[1;33m[ UNKOWN ]\033[0m No scatter or aggregate phase");
    }

    if (vecTimeAggregate.size() < 2 * numLayers) {
        for (unsigned i = vecTimeAggregate.size(); i < 2 * numLayers; i++) {
            vecTimeAggregate.push_back(0.0);
        }
    }
    vecTimeAggregate[numLayers + iteration] += getTimer() - sttTimer;

    backwardGhostVerticesDataIn = backwardGhostVerticesDataOut;

    return outputTensor;
}


//////////////////////////////////////////////////////////
// Below are private backward functions for the engine. //
//////////////////////////////////////////////////////////
// Backward scatter phase functions
FeatType* Engine::applyScatterPhase(FeatType* gradTensor, unsigned vtcsCnt,
  unsigned inFeatDim, unsigned outFeatDim, bool scatter) {
    double sttTimer = getTimer();

    assert(vtcsCnt == graph.localVtxCnt);
    commHalt = false;

    FeatType *outputTensor = new FeatType[vtcsCnt * inFeatDim];
    auto bgr_fp = std::bind(&Engine::backwardGhostReceiver, this,
                    std::placeholders::_1, std::placeholders::_2);
    auto bgu_fp = std::bind(&Engine::pipelineBackwardGhostGradients, this,
                    std::placeholders::_1, std::placeholders::_2);
    std::thread scatterThread;
    if (scatter) {
        backwardGhostVerticesDataOut = new FeatType[graph.dstGhostCnt * inFeatDim];
        dataPool->perform(bgr_fp, (void*) &inFeatDim);
        scatterThread = std::thread(bgu_fp, outputTensor, inFeatDim);
    }

    resComm->newContextBackward(iteration - 1, gradTensor, outputTensor,
      savedTensors, localVerticesLabels, vtcsCnt, inFeatDim, outFeatDim,
      getFeatDim(numLayers), scatter);
    resComm->requestBackward(iteration - 1, iteration - 1 == numLayers - 1);

    if (scatter) {
        scatterThread.join();
    }
    commHalt = true;
    dataPool->sync();

    delete[] gradTensor;
    for (auto &sTensor : savedTensors[iteration - 1]) {
        delete[] sTensor.getData();
    }
    savedTensors[iteration - 1].clear();

    if (vecTimeLambda.size() < 2 * numLayers) {
        for (unsigned i = vecTimeLambda.size(); i < 2 * numLayers; i++) {
            vecTimeLambda.push_back(0.0);
        }
    }
    vecTimeLambda[numLayers + iteration - 1] += getTimer() - sttTimer;

    return outputTensor;
}

FeatType* Engine::aggregateApplyScatterPhase(FeatType* gradTensor, unsigned vtcsCnt,
  unsigned inFeatDim, unsigned outFeatDim, bool scatter) {
    // Prepare for gather phase
    FeatType *gatheredTensor = new FeatType[vtcsCnt * outFeatDim];
    FeatType *outputTensor = new FeatType[vtcsCnt * inFeatDim];
    auto bgr_fp = std::bind(&Engine::backwardGhostReceiver, this,
                    std::placeholders::_1, std::placeholders::_2);
    auto bgu_fp = std::bind(&Engine::pipelineBackwardGhostGradients, this,
                    std::placeholders::_1, std::placeholders::_2);
    commHalt = false;
    bkwdRecvCnt = 0;
    std::thread scatterThread;
    if (scatter) {
        backwardGhostVerticesDataOut = new FeatType[graph.dstGhostCnt * outFeatDim];
        dataPool->perform(bgr_fp, (void*) &inFeatDim);
        scatterThread = std::thread(bgu_fp, outputTensor, outFeatDim);
    }

    currId = 0;

    // Start gathering
    AggOPArgs args = {gatheredTensor, gradTensor, vtcsCnt, outFeatDim};
    auto computeFn = std::bind(&Engine::aggregateBPCompute, this, std::placeholders::_1, std::placeholders::_2);
    computePool->perform(computeFn, &args);

    // Prepare for applyVertex phase
    resComm->newContextBackward(iteration - 1, gatheredTensor, outputTensor, savedTensors, localVerticesLabels, vtcsCnt, inFeatDim, outFeatDim, getFeatDim(numLayers), scatter);

    // Start applyVertex phase
    unsigned currLambdaId = 0;
    if (mode == LAMBDA) {
        const unsigned lambdaChunkSize = (vtcsCnt + numLambdasForward - 1) / numLambdasBackward;
        unsigned availChunkSize = lambdaChunkSize;
        while (currId < vtcsCnt) {
            unsigned lvid = currId;
            if (lvid > availChunkSize) {
                resComm->invokeLambdaBackward(iteration - 1, currLambdaId, iteration - 1 == numLayers - 1);
                availChunkSize += lambdaChunkSize;
                ++currLambdaId;
            }
            usleep(2000); // wait for 2ms and check again
        }
    }
    computePool->sync();
    if (mode != LAMBDA) {
        resComm->requestBackward(iteration - 1, iteration - 1 == numLayers - 1);
    } else {
        while (currLambdaId < numLambdasBackward) {
            resComm->invokeLambdaBackward(iteration - 1, currLambdaId, iteration - 1 == numLayers - 1);
            ++currLambdaId;
        }
        resComm->waitLambdaBackward(iteration - 1, iteration - 1 == numLayers - 1);
    }

    if (scatter) {
        scatterThread.join();
    }
    commHalt = true;
    dataPool->sync();

    // Clean up applyVertex phase
    delete[] gatheredTensor;
    for (auto &sTensor : savedTensors[iteration - 1]) {
        delete[] sTensor.getData();
    }
    savedTensors[iteration - 1].clear();

    // Clean up gather phase
    delete[] gradTensor;
    delete[] backwardGhostVerticesDataIn;

    return outputTensor;
}

FeatType* Engine::aggregateApplyPhase(FeatType* gradTensor, unsigned vtcsCnt,
  unsigned inFeatDim, unsigned outFeatDim, bool scatter) {
    double sttTimer = getTimer();

    // Prepare for gather phase
    FeatType *gatheredTensor = new FeatType[vtcsCnt * outFeatDim];
    FeatType *outputTensor = new FeatType[vtcsCnt * inFeatDim];
    currId = 0;

    // Start gathering
    AggOPArgs args = {gatheredTensor, gradTensor, vtcsCnt, outFeatDim};
    auto computeFn = std::bind(&Engine::aggregateBPCompute, this, std::placeholders::_1, std::placeholders::_2);
    computePool->perform(computeFn, &args);

    // Prepare for applyVertex phase
    resComm->newContextBackward(iteration - 1, gatheredTensor, outputTensor, savedTensors, localVerticesLabels, vtcsCnt, inFeatDim, outFeatDim, getFeatDim(numLayers));

    // Start applyVertex phase
    unsigned currLambdaId = 0;
    if (mode == LAMBDA) {
        const unsigned lambdaChunkSize = (vtcsCnt + numLambdasForward - 1) / numLambdasBackward;
        unsigned availChunkSize = lambdaChunkSize;
        while (currId < vtcsCnt) {
            unsigned lvid = currId;
            if (lvid > availChunkSize) {
                resComm->invokeLambdaBackward(iteration - 1, currLambdaId, iteration - 1 == numLayers - 1);
                availChunkSize += lambdaChunkSize;
                ++currLambdaId;
            }
            usleep(2000); // wait for 2ms and check again
        }
    }
    computePool->sync();
    if (mode != LAMBDA) {
        resComm->requestBackward(iteration - 1, iteration - 1 == numLayers - 1);
    } else {
        while (currLambdaId < numLambdasBackward) {
            resComm->invokeLambdaBackward(iteration - 1, currLambdaId, iteration - 1 == numLayers - 1);
            ++currLambdaId;
        }
        resComm->waitLambdaBackward(iteration - 1, iteration - 1 == numLayers - 1);
    }

    // Clean up applyVertex phase
    delete[] gatheredTensor;
    for (auto &sTensor : savedTensors[iteration - 1]) {
        delete[] sTensor.getData();
    }
    savedTensors[iteration - 1].clear();

    // Clean up gather phase
    delete[] gradTensor;
    delete[] backwardGhostVerticesDataIn;

    if (vecTimeAggregate.size() < 2 * numLayers) {
        for (unsigned i = vecTimeAggregate.size(); i < 2 * numLayers; i++) {
            vecTimeAggregate.push_back(0.0);
        }
    }
    vecTimeAggregate[numLayers + iteration] += getTimer() - sttTimer;

    return outputTensor;
}

void Engine::aggregateBPCompute(unsigned tid, void *args) {
    FeatType *nextGradTensor = ((AggOPArgs *) args)->outputTensor;
    FeatType *gradTensor = ((AggOPArgs *) args)->inputTensor;
    const unsigned vtcsCnt = ((AggOPArgs *) args)->vtcsCnt;
    const unsigned featDim = ((AggOPArgs *) args)->featDim;

    unsigned lvid = 0;
    while (currId < vtcsCnt) {
        lvid = __sync_fetch_and_add(&currId, 1);
        if (lvid < vtcsCnt) {
            backwardAggregateFromNeighbors(lvid, nextGradTensor, gradTensor, featDim);
        }
    }
}


/**
 *
 * Aggregate featDim feature values starting from offset from all neighbors (including self). Then write the results to the
 * data buffer area for serialization. The results are to be used for being sent to lambda threads.
 *
 */
void
Engine::backwardAggregateFromNeighbors(unsigned lvid, FeatType *nextGradTensor, FeatType *gradTensor, unsigned featDim) {

    // Read out data of the current iteration of given vertex.
    FeatType *currDataDst = getVtxFeat(nextGradTensor, lvid, featDim);
    FeatType *currDataPtr = getVtxFeat(gradTensor, lvid, featDim);

    memcpy(currDataDst, currDataPtr, featDim * sizeof(FeatType));

    // Apply normalization factor on the current data.
    {
        const EdgeType normFactor = graph.vtxDataVec[lvid];
        for (unsigned i = 0; i < featDim; ++i) {
            currDataDst[i] *= normFactor;
        }
    }

    // Aggregate from incoming neighbors.
    for (unsigned eid = graph.backwardAdj.rowPtrs[lvid]; eid < graph.backwardAdj.rowPtrs[lvid + 1]; ++eid) {
        FeatType *otherDataPtr;
        EdgeType normFactor = graph.backwardAdj.values[eid];
        unsigned dstVId = graph.backwardAdj.columnIdxs[eid];
        if (dstVId < graph.localVtxCnt) { // local vertex.
            otherDataPtr = getVtxFeat(gradTensor, dstVId, featDim);
        } else {                          // ghost vertex.
            otherDataPtr = getVtxFeat(backwardGhostVerticesDataIn, dstVId - graph.localVtxCnt, featDim);
        }
        for (unsigned j = 0; j < featDim; ++j) {
            currDataDst[j] += otherDataPtr[j] * normFactor;
        }
    }
}

void
Engine::sendBackwardGhostGradients(FeatType *gradTensor, unsigned featDim) {
    // Loop through all local vertices and do the data send out work. If there are any remote edges for a vertex, should send this vid to
    // other nodes for their ghost's update.
    bool batchFlag = false;
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
            bkwdRecvCntLock.lock();
            bkwdRecvCnt++;
            bkwdRecvCntLock.unlock();
        }
    }
    // Wait for all remote schedulings sent by me to be handled.
    bkwdRecvCntLock.lock();
    if (bkwdRecvCnt > 0) {
        bkwdRecvCntCond.wait();
    }
    bkwdRecvCntLock.unlock();
}

inline void
Engine::pipelineBackwardGhostGradients(FeatType* inputTensor, unsigned featDim) {
    int failedTrials = 0;
    const int INIT_PERIOD = 256;
    const int MAX_PERIOD = 4096;
    int SLEEP_PERIOD = INIT_PERIOD;
    unsigned partsScattered = 0;

    partsScatteredTable = new bool[numLambdasBackward];
    std::memset(partsScatteredTable, 0, sizeof(bool) * numLambdasBackward);

    // Check queue to see if partition ready
    while (partsScattered < numLambdasBackward) {
        consumerQueueLock.lock();
        if (rangesToScatter.empty()) {
            consumerQueueLock.unlock();
            // sleep with backoff
            usleep(SLEEP_PERIOD); // sleep a little and give up CPUs
            failedTrials++;
            if (failedTrials == 64 && SLEEP_PERIOD < MAX_PERIOD) {
                failedTrials = 0;
                SLEEP_PERIOD *= 2;
            }

        } else {
            std::pair<unsigned, unsigned> partitionInfo = rangesToScatter.front();
            rangesToScatter.pop();
            // Has this partition already been processed
            if (partsScatteredTable[partitionInfo.first]) {
                consumerQueueLock.unlock();
                continue;
            }
            partsScatteredTable[partitionInfo.first] = true;
            consumerQueueLock.unlock();

            // Partition Info: (partId, rowsPerPartition)
            unsigned startId = partitionInfo.first * partitionInfo.second;
            unsigned endId = (partitionInfo.first + 1) * partitionInfo.second;
            endId = endId > graph.localVtxCnt ? graph.localVtxCnt : endId;



            // Create a series of buckets for batching sendout messages to nodes
            std::vector<unsigned>* batchedIds = new std::vector<unsigned>[numNodes];
            for (unsigned lvid = startId; lvid < endId; ++lvid) {
                for (unsigned nid : graph.backwardGhostMap[lvid]) {
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

                unsigned backwardGhostVCnt = batchedIds[nid].size();
                for (unsigned ib = 0; ib < backwardGhostVCnt; ib += BATCH_SIZE) {
                    unsigned sendBatchSize = (backwardGhostVCnt - ib) < BATCH_SIZE ? (backwardGhostVCnt - ib) : BATCH_SIZE;

                    backwardVerticesPushOut(nid, sendBatchSize, batchedIds[nid].data() + ib, inputTensor, featDim);
                    bkwdRecvCntLock.lock();
                    bkwdRecvCnt++;
                    bkwdRecvCntLock.unlock();
                }
            }

            delete[] batchedIds;
            failedTrials = 0;
            partsScattered++;
        }
    }

    // Once all partitions scattered, wait on all acks
    bkwdRecvCntLock.lock();
    if (bkwdRecvCnt > 0) {
        bkwdRecvCntCond.wait();
    }
    bkwdRecvCntLock.unlock();
}

inline void
Engine::backwardVerticesPushOut(unsigned receiver, unsigned totCnt, unsigned *lvids, FeatType *gradTensor, unsigned featDim) {
    zmq::message_t msg(DATA_HEADER_SIZE + (sizeof(unsigned) + sizeof(FeatType) * featDim) * totCnt);
    char *msgPtr = (char *)(msg.data());
    sprintf(msgPtr, NODE_ID_HEADER, receiver);
    msgPtr += NODE_ID_DIGITS;
    *(unsigned *)msgPtr = nodeId;
    msgPtr += sizeof(unsigned);
    *(unsigned *)msgPtr = totCnt;
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
Engine::backwardGhostReceiver(unsigned tid, void* _featDim) {
    // backoff sleep strategy to improve CPU utilization
    int failedTrials = 0;
    const int INIT_PERIOD = 256;
    const int MAX_PERIOD = 4096;
    int SLEEP_PERIOD = INIT_PERIOD;
    unsigned sender, topic;
    unsigned vtcsRecvd = 0;
    unsigned featDim = (*(unsigned*)_featDim);
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
                // waiting on it to be empty at the iteration barrier.
            } else { // (topic == MAX_IDTYPE - 1)
                bkwdRecvCntLock.lock();
                bkwdRecvCnt--;
                bkwdRecvCntLock.unlock();
            }
            bkwdRecvCntLock.lock();
            if (bkwdRecvCnt == 0 && vtcsRecvd == graph.dstGhostCnt) {
                bkwdRecvCntCond.signal();
            }
            bkwdRecvCntLock.unlock();

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
    savedVtxTensors.emplace(name, Matrix(name.c_str(), rows, cols, dptr));
}

void Engine::saveTensor(const char* name, unsigned rows, unsigned cols, FeatType* dptr) {
    auto iter = savedVtxTensors.find(name);
    if (iter != savedVtxTensors.end()) {
        delete[] (iter->second).getData();
        savedVtxTensors.erase(iter);
    }
    savedVtxTensors.emplace(std::string(name), Matrix(name, rows, cols, dptr));
}

void Engine::saveTensor(Matrix& mat) {
    auto iter = savedVtxTensors.find(mat.name());
    if (iter != savedVtxTensors.end()) {
        delete[] (iter->second).getData();
        savedVtxTensors.erase(iter);
    }
    savedVtxTensors.emplace(mat.name(), mat);
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
            printLog(nodeId, "<EM>    Lambda        %2u  %.3lf ms", i, vecTimeLambda[i] / (float)numEpochs);
            printLog(nodeId, "<EM>    Ghost update  %2u  %.3lf ms", i, vecTimeSendout[i] / (float)numEpochs);
        }
    }
    printLog(nodeId, "<EM>: Average forward-prop time %.3lf ms", timeForwardProcess / (float)numEpochs);

    if (!pipeline) {
        printLog(nodeId, "<EM>: Backward: Average time per stage:");
        for (unsigned i = numLayers; i < 2 * numLayers; i++) {
            printLog(nodeId, "<EM>    Aggregation   %2u  %.3lf ms", i, vecTimeAggregate[i] / (float)numEpochs);
            printLog(nodeId, "<EM>    Lambda        %2u  %.3lf ms", i, vecTimeLambda[i] / (float)numEpochs);
            printLog(nodeId, "<EM>    Ghost update  %2u  %.3lf ms", i, vecTimeSendout[i] / (float)numEpochs);
        }
    }
    printLog(nodeId, "<EM>: Backward-prop takes %.3lf ms", timeBackwardProcess / (float)numEpochs);

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
    ("pipeline", boost::program_options::value<bool>(), "0: Sequential, 1: Pipelined");
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

Engine engine;
