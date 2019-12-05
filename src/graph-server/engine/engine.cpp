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
#include "engine.hpp"

#ifdef _GPU_ENABLED_
#include "../GPU-Computation/comp_unit.cuh"
#endif

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

    // Leave buf for myself empty, which is a little wasting.
    forwardGhostsList = new std::vector<unsigned> [numNodes];
    backwardGhostsList = new std::vector<unsigned> [numNodes];

    // Read in the graph and subscribe vertex global ID topics.
    readGraphBS(graphFile);
    printGraphMetrics();

    // save intermediate tensors during forward phase for backward computation.
    savedTensors = new std::vector<Matrix> [numLayers];

    // Init it here for collecting data when reading files
    forwardVerticesInitData = new FeatType[getFeatDim(0) * graph.getNumLocalVertices()];
    forwardGhostInitData = new FeatType[getFeatDim(0) * graph.getNumInEdgeGhostVertices()];
    // Create labels storage area. Read in labels and store as one-hot format.
    localVerticesLabels = new FeatType[layerConfig[numLayers] * graph.getNumLocalVertices()];


    // Set a local index for all ghost vertices along the way. This index is used for indexing within the ghost data arrays.
    unsigned ghostCount = 0;

    for (auto it = graph.getInEdgeGhostVertices().begin(); it != graph.getInEdgeGhostVertices().end(); it++) {
        it->second.setLocalId(ghostCount++);
    }
    ghostCount = 0;
    for (auto it = graph.getOutEdgeGhostVertices().begin(); it != graph.getOutEdgeGhostVertices().end(); it++) {
        it->second.setLocalId(ghostCount++);
    }


    // Read in initial feature values (input features) & labels.
    readFeaturesFile(featuresFile);
    readLabelsFile(labelsFile);

    // Initialize synchronization utilities.
    recvCnt = 0;
    lockRecvCnt.init();
    condRecvCnt.init(lockRecvCnt);

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

    // Compact the graph.
    graph.compactGraph();

    setUpCommInfo();
    resComm = createResourceComm(gpuEnabled ? "GPU" : "Lambda", commInfo);

    timeForwardProcess = 0.0;
    timeInit += getTimer();
    printLog(nodeId, "Engine initialization complete.");
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

    lockRecvCnt.destroy();
    condRecvCnt.destroy();

    resComm->sendShutdownMessage();

    destroyResourceComm(gpuEnabled ? "GPU" : "Lambda", resComm);

    delete[] forwardGhostsList;
    delete[] backwardGhostsList;

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
Engine::setUpCommInfo() {
    commInfo.nodeIp = nodeManager.getNode(nodeId).pubip;
    commInfo.nodeId = nodeId;
    commInfo.dataserverPort = dataserverPort;
    commInfo.coordserverIp = coordserverIp;
    commInfo.coordserverPort = coordserverPort;
    commInfo.numLambdasForward = numLambdasForward;
    commInfo.numLambdasBackward = numLambdasBackward;
    commInfo.numNodes = numNodes;
    commInfo.wServersFile = weightserverIPFile;
    commInfo.weightserverPort = weightserverPort;
    commInfo.totalLayers = numLayers;
}

/**
 *
 * Set the training validation split based on the partitions
 * float trainPortion must be between (0,1)
 *
 */
void
Engine::setTrainValidationSplit(float trainPortion) {
    resComm->setTrainValidationSplit(trainPortion, graph.getNumLocalVertices());
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

/**
 *
 * Whether GPU is enabled or not
 *
 */
bool
Engine::isGPUEnabled() {
    return Engine::gpuEnabled;
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
    return Engine::numEpochs;
}

/**
 *
 * How many epochs to run before validation
 *
 */
unsigned
Engine::getValFreq() {
    return Engine::valFreq;
}

/**
 *
 * Return the ID of this node
 *
 */
unsigned
Engine::getNodeId() {
    return Engine::nodeId;
}

/**
 *
 * Runs a forward propagation phase: (Aggregate -> Lambda Computing -> Ghost Update) -> ( ... ) -> ...
 * Will start a bunch of worker threads and a bunch of data communicator threads.
 *
 */
FeatType *
Engine::runForward(bool eval) {
    // Make sure all nodes start running the forward-prop phase.
    nodeManager.barrier();
    printLog(nodeId, "Engine starts running FORWARD...");

    timeForwardProcess -= getTimer();

    evaluate = false;

    const unsigned graphLocalVerticesNum = graph.getNumLocalVertices();
    // Create buffer for first-layer aggregation.
    FeatType *inputTensor = forwardVerticesInitData;
    forwardGhostVerticesData = forwardGhostInitData;
    for (iteration = 0; iteration < numLayers; iteration++) {
        inputTensor = aggregate(inputTensor, graphLocalVerticesNum, getFeatDim(iteration));
        inputTensor = invokeLambda(inputTensor, graphLocalVerticesNum, getFeatDim(iteration), getFeatDim(iteration + 1));
        if (iteration < numLayers - 1) { // don't need scatter at the last layer.
            inputTensor = scatter(inputTensor, graphLocalVerticesNum, getFeatDim(iteration + 1));
        }
    }

    timeForwardProcess += getTimer();
    printLog(nodeId, "Engine completes FORWARD at iter %u.", iteration);
    calcAcc(inputTensor, localVerticesLabels, graphLocalVerticesNum, getFeatDim(numLayers));

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
    nodeManager.barrier();
    printLog(nodeId, "Engine starts running BACKWARD...");

    timeBackwardProcess -= getTimer();

    const unsigned graphLocalVerticesNum = graph.getNumLocalVertices();
    // Create buffer for first-layer aggregation.
    FeatType *gradTensor = initGradTensor;
    for (iteration = numLayers; iteration > 0; iteration--) {
        gradTensor = invokeLambdaBackward(gradTensor, graphLocalVerticesNum, getFeatDim(iteration - 1), getFeatDim(iteration));
        if (iteration > 1) {
            gradTensor = scatterBackward(gradTensor, graphLocalVerticesNum, getFeatDim(iteration - 1));
            gradTensor = aggregateBackward(gradTensor, graphLocalVerticesNum, getFeatDim(iteration - 1));
        }
    }

    delete[] gradTensor;

    timeBackwardProcess += getTimer();
    printLog(nodeId, "Engine completes BACKWARD at iter %u.", iteration);
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

    char outBuf[1024];
    sprintf(outBuf, "<EM>: Initialization takes %.3lf ms", timeInit);
    outStream << outBuf << std::endl;
    sprintf(outBuf, "<EM>: Forward:  Time per stage:");
    outStream << outBuf << std::endl;
    for (unsigned i = 0; i < numLayers; ++i) {
        sprintf(outBuf, "<EM>    Aggregation   %2u  %.3lf ms", i, vecTimeAggregate[i]);
        outStream << outBuf << std::endl;
        sprintf(outBuf, "<EM>    Lambda        %2u  %.3lf ms", i, vecTimeLambda[i]);
        outStream << outBuf << std::endl;
        sprintf(outBuf, "<EM>    Ghost update  %2u  %.3lf ms", i, vecTimeSendout[i]);
        outStream << outBuf << std::endl;
    }
    sprintf(outBuf, "<EM>: Total forward-prop time %.3lf ms", timeForwardProcess);
    outStream << outBuf << std::endl;

    sprintf(outBuf, "<EM>: Backward: Time per stage:");
    outStream << outBuf << std::endl;
    for (unsigned i = numLayers; i < 2 * numLayers; i++) {
        sprintf(outBuf, "<EM>    Aggregation   %2u  %.3lf ms", i, vecTimeAggregate[i]);
        outStream << outBuf << std::endl;
        sprintf(outBuf, "<EM>    Lambda        %2u  %.3lf ms", i, vecTimeLambda[i]);
        outStream << outBuf << std::endl;
        sprintf(outBuf, "<EM>    Ghost update  %2u  %.3lf ms", i, vecTimeSendout[i]);
        outStream << outBuf << std::endl;
    }
    sprintf(outBuf, "<EM>: Backward-prop takes %.3lf ms", timeBackwardProcess);
    outStream << outBuf << std::endl;

    // Write benchmarking results to log file.
    if (master()) {
        assert(vecTimeAggregate.size() == 2 * numLayers);
        assert(vecTimeLambda.size() == 2 * numLayers);
        assert(vecTimeSendout.size() == 2 * numLayers);
        printEngineMetrics();
    }
}


struct AggOPArgs {
    FeatType *outputTensor;
    FeatType *inputTensor;
    unsigned vtcsCnt;
    unsigned featDim;
};


#ifndef _GPU_ENABLED_
static CuMatrix *NormAdjMatrixIn = NULL;
FeatType *Engine::aggregate(FeatType *vtcsTensor, unsigned vtcsCnt, unsigned featDim) {
    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < vtcsCnt*featDim; ++i)
    {
        if(i%2000000==0)
            std::cout<<vtcsTensor[i]<<" ";
    }
    std::cout<<std::endl;
    ComputingUnit cu = ComputingUnit::getInstance();
    if (NormAdjMatrixIn == NULL) {
        NormAdjMatrixIn = new CuMatrix();
        NormAdjMatrixIn->loadSpCsrForward(cu.spHandle,
                                          graph.getNumLocalVertices(),
                                          graph.getVertices(),
                                          graph.getNumInEdgeGhostVertices());
        auto t3 = std::chrono::high_resolution_clock::now();
        std::cout << "ADJ build "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t1).count()
                  << " milliseconds\n";
    }

    double sttTimer = getTimer();
    FeatType *outputTensor = new FeatType [(vtcsCnt) * featDim];
    CuMatrix feat;
    auto t7 = std::chrono::high_resolution_clock::now();
    feat.loadSpDense(vtcsTensor, forwardGhostVerticesData,
                     graph.getNumLocalVertices(), graph.getNumInEdgeGhostVertices(),
                     featDim);
    auto t5 = std::chrono::high_resolution_clock::now();
    std::cout << "loadSpDense[CU] "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t5 - t7).count()
              << " milliseconds\n";
    CuMatrix out = cu.aggregate(*NormAdjMatrixIn, feat);
    auto t4 = std::chrono::high_resolution_clock::now();
    std::cout << "aggregate[CU] "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t5).count()
              << " milliseconds\n";
    out = out.transpose();
    cudaDeviceSynchronize();
    auto t6 = std::chrono::high_resolution_clock::now();
    std::cout << "transpose "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t6 - t4).count()
              << " milliseconds\n";
    out.setData(outputTensor);
    out.updateMatrixFromGPU();

    currId = vtcsCnt;
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "Aggregate[TOTAL] "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
              << " milliseconds\n";


    if (iteration > 0) {
        delete[] forwardGhostVerticesData;
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
    auto t1 = std::chrono::high_resolution_clock::now();
    FeatType *outputTensor = new FeatType [vtcsCnt * featDim];
    currId = 0;

    AggOPArgs args = {outputTensor, vtcsTensor, vtcsCnt, featDim};
    auto computeFn = std::bind(&Engine::aggregateCompute, this, std::placeholders::_1, std::placeholders::_2);

    computePool->perform(computeFn, &args);
    computePool->sync();

    if (iteration > 0) {
        delete[] forwardGhostVerticesData;
        delete[] vtcsTensor;
    }


    if (vecTimeAggregate.size() < numLayers) {
        vecTimeAggregate.push_back(getTimer() - sttTimer);
    } else {
        vecTimeAggregate[iteration] += getTimer() - sttTimer;
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "Aggregate "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
              << " milliseconds\n";
    Matrix m(vtcsCnt, featDim, outputTensor);
    return outputTensor;
}
#endif // _GPU_ENABLED_

FeatType *
Engine::invokeLambda(FeatType *vtcsTensor, unsigned vtcsCnt, unsigned inFeatDim, unsigned outFeatDim) {
    double sttTimer = getTimer();
    assert(vtcsCnt == graph.getNumLocalVertices());

    FeatType *outputTensor = new FeatType [vtcsCnt * outFeatDim];
    FeatType *zTensor = new FeatType [vtcsCnt * outFeatDim];

    std::vector<FeatType *> savedTensorBufs;
    bool saveInput = true;
    if (saveInput) {
        savedTensors[iteration].push_back(Matrix(vtcsCnt, inFeatDim, vtcsTensor));
    }
    savedTensors[iteration].push_back(Matrix(vtcsCnt, outFeatDim, zTensor));

    bool runEval = evaluate && iteration == numLayers - 1;
    // Start a new lambda communication context.
    resComm->newContextForward(iteration, vtcsTensor, zTensor, outputTensor, vtcsCnt, inFeatDim, outFeatDim, runEval);

    // if in GPU mode we launch gpu computation here and wait the results
    if (gpuEnabled) {
        resComm->requestForward(iteration, iteration == numLayers - 1);
    } else {
        unsigned availLambdaId = 0;
        while(availLambdaId < numLambdasForward) {
            resComm->invokeLambdaForward(iteration, availLambdaId, iteration == numLayers - 1);
            availLambdaId++;
        }
        resComm->waitLambdaForward(iteration, iteration == numLayers - 1);
    }

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
    // printLog(nodeId, "All lambda requests finished. Results received.");

    return outputTensor;
}

FeatType *
Engine::scatter(FeatType *vtcsTensor, unsigned vtcsCnt, unsigned featDim) {
    double sttTimer = getTimer();

    // Start data communicators.
    commHalt = false;
    forwardGhostVerticesData = new FeatType [featDim * graph.getNumInEdgeGhostVertices()];
    auto fgc_fp = std::bind(&Engine::forwardGhostCommunicator, this, std::placeholders::_1, std::placeholders::_2);
    dataPool->perform(fgc_fp);

    sendForwardGhostUpdates(vtcsTensor, featDim);

    // TODO: (YIFAN) we can optimize this to extend comm protocal. Mark the last packet sent so this node knows when to exit ghostCommunicator.
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


/////////////////////////////////////////////////
// Below are private functions for the engine. //
/////////////////////////////////////////////////
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
    Vertex &v = graph.getVertex(lvid);
    for (unsigned i = 0; i < featDim; ++i) {
        currDataDst[i] *= v.getNormFactor();
    }

    // Aggregate from incoming neighbors.
    for (unsigned i = 0; i < v.getNumInEdges(); ++i) {
        FeatType *otherDataPtr;
        EdgeType normFactor = v.getInEdge(i).getData();

        if (v.getInEdge(i).getEdgeLocation() == LOCAL_EDGE_TYPE) {    // Local vertex.
            otherDataPtr = getVtxFeat(inputTensor, v.getSourceVertexLocalId(i), featDim);
        } else {                                                      // Ghost vertex.
            otherDataPtr = getVtxFeat(forwardGhostVerticesData, v.getSourceVertexLocalId(i), featDim);
        }
        // TODO: Locks on the data array area is not properly set yet. But does not affect forward prop.
        for (unsigned j = 0; j < featDim; ++j) {
            currDataDst[j] += otherDataPtr[j] * normFactor;
        }
    }

}

inline void
Engine::sendForwardGhostUpdates(FeatType *inputTensor, unsigned featDim) {
    // Loop through all local vertices and do the data send out work. If there are any remote edges for a vertex, should send this vid to
    // other nodes for their ghost's update.
    // TODO: (YIFAN) process crashes when return if BATCH_SIZE is too large. Weird, to be fixed.
    // Please decrease BATCH_SIZE if porcess crashed.
    bool batchFlag = true;
    unsigned BATCH_SIZE = std::max(((batchFlag ? MAX_MSG_SIZE : 4096) - DATA_HEADER_SIZE) /
                                   (sizeof(unsigned) + sizeof(FeatType) * featDim), 1ul); // at least send one vertex
    for (unsigned nid = 0; nid < numNodes; ++nid) {
        if (nid == nodeId) {
            continue;
        }
        unsigned forwardGhostVCnt = forwardGhostsList[nid].size();
        for (unsigned ib = 0; ib < forwardGhostVCnt; ib += BATCH_SIZE) {
            unsigned sendBatchSize = (forwardGhostVCnt - ib) < BATCH_SIZE ? (forwardGhostVCnt - ib) : BATCH_SIZE;

            forwardVerticesPushOut(nid, sendBatchSize, forwardGhostsList[nid].data() + ib, inputTensor, featDim);
            lockRecvCnt.lock();
            recvCnt++;
            lockRecvCnt.unlock();
        }
    }
    // Wait for all remote schedulings sent by me to be handled.
    lockRecvCnt.lock();
    if (recvCnt > 0) {
        condRecvCnt.wait();
    }
    lockRecvCnt.unlock();
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
 * Major part of the engine's communication logic is done by data threads. These threads loop asynchronously with computation workers.
 *
 */
void
Engine::forwardGhostCommunicator(unsigned tid, void *args) {
    // backoff sleep strategy to improve CPU utilization
    int failedTrials = 0;
    const int INIT_PERIOD = 256;
    const int MAX_PERIOD = 4096;
    int SLEEP_PERIOD = INIT_PERIOD;
    unsigned sender, topic;
    FeatType *msgBuf = (FeatType *)new char[MAX_MSG_SIZE];

    // While loop, looping infinitely to get the next message.
    while (!commHalt) {
        // No message in queue.
        if (!commManager.dataPullIn(&sender, &topic, msgBuf, MAX_MSG_SIZE)) {
            // Computation workers done their work, so communicator goes to death as well.
            if (commHalt) {
                break;
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
                unsigned featDim = getFeatDim(iteration + 1);
                // Update ghost vertices
                for (unsigned i = 0; i < recvGhostVCnt; ++i) {
                    unsigned gvid = *(unsigned *)bufPtr;
                    bufPtr += sizeof(unsigned);
                    FeatType *dataPtr = getVtxFeat(forwardGhostVerticesData, graph.getInEdgeGhostVertex(gvid).getLocalId(), featDim);
                    memcpy(dataPtr, bufPtr, sizeof(FeatType) * featDim);
                    bufPtr += sizeof(FeatType) * featDim;
                }

                // A respond to a broadcast, and the topic vertex is in my local vertices. I should update the
                // corresponding recvWaiter's value. If waiters become empty, send a signal in case the workers are
                // waiting on it to be empty at the iteration barrier.
            } else { // (topic == MAX_IDTYPE - 1)
                lockRecvCnt.lock();
                recvCnt--;
                if (recvCnt == 0) {
                    condRecvCnt.signal();
                }
                lockRecvCnt.unlock();
            }

            SLEEP_PERIOD = INIT_PERIOD;
            failedTrials = 0;
        }
    }
    delete[] msgBuf;
}


/**
 *
 * Major part of the engine's backward-prop logic.
 *
 */
#ifdef _GPU_ENABLED_
static CuMatrix *NormAdjMatrixOut = NULL;
FeatType *
Engine::aggregateBackward(FeatType *gradTensor, unsigned vtcsCnt, unsigned featDim) {
    double sttTimer = getTimer();
    auto t1 = std::chrono::high_resolution_clock::now();
    currId = 0;
    FeatType *outputTensor = new FeatType [vtcsCnt * featDim];

    ComputingUnit cu = ComputingUnit::getInstance();
    if (NormAdjMatrixOut == NULL) {
        NormAdjMatrixOut = new CuMatrix();
        NormAdjMatrixOut->loadSpCsrBackward(cu.spHandle,
                                            graph.getNumLocalVertices(),
                                            graph.getVertices(),
                                            graph.getNumOutEdgeGhostVertices());
        auto t3 = std::chrono::high_resolution_clock::now();
        std::cout << "ADJ[Backward] build "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t1).count()
                  << " milliseconds\n";
    }

    CuMatrix feat;
    auto t7 = std::chrono::high_resolution_clock::now();
    feat.loadSpDense(gradTensor, backwardGhostVerticesData,
                     graph.getNumLocalVertices(), graph.getNumOutEdgeGhostVertices(),
                     featDim);

    auto t5 = std::chrono::high_resolution_clock::now();
    std::cout << "loadSpDense[Backward] "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t5 - t7).count()
              << " milliseconds\n";
    CuMatrix out = cu.aggregate(*NormAdjMatrixOut, feat);
    auto t4 = std::chrono::high_resolution_clock::now();
    std::cout << "aggregate[Backward] "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t5).count()
              << " milliseconds\n";
    out = out.transpose();
    auto t6 = std::chrono::high_resolution_clock::now();
    std::cout << "transpose[Backward] "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t6 - t4).count()
              << " milliseconds\n";
    out.setData(outputTensor);
    out.updateMatrixFromGPU();

    currId = vtcsCnt;

    delete[] gradTensor;
    delete[] backwardGhostVerticesData;

    if (vecTimeAggregate.size() < 2 * numLayers) {
        for (unsigned i = vecTimeAggregate.size(); i < 2 * numLayers; i++) {
            vecTimeAggregate.push_back(0.0);
        }
    }
    vecTimeAggregate[numLayers + iteration - 1] += getTimer() - sttTimer;

    return outputTensor;
}
#else
FeatType *
Engine::aggregateBackward(FeatType *gradTensor, unsigned vtcsCnt, unsigned featDim) {
    double sttTimer = getTimer();

    FeatType *outputTensor = new FeatType [vtcsCnt * featDim];
    currId = 0;

    AggOPArgs args = {outputTensor, gradTensor, vtcsCnt, featDim};
    auto computeFn = std::bind(&Engine::aggregateBPCompute, this, std::placeholders::_1, std::placeholders::_2);
    computePool->perform(computeFn, &args);
    computePool->sync();

    delete[] gradTensor;
    delete[] backwardGhostVerticesData;

    if (vecTimeAggregate.size() < 2 * numLayers) {
        for (unsigned i = vecTimeAggregate.size(); i < 2 * numLayers; i++) {
            vecTimeAggregate.push_back(0.0);
        }
    }
    vecTimeAggregate[numLayers + iteration - 1] += getTimer() - sttTimer;

    return outputTensor;
}
#endif



FeatType *
Engine::invokeLambdaBackward(FeatType *gradTensor, unsigned vtcsCnt, unsigned inFeatDim, unsigned outFeatDim) {
    double sttTimer = getTimer();

    assert(vtcsCnt == graph.getNumLocalVertices());

    FeatType *outputTensor = new FeatType [vtcsCnt * inFeatDim];

    resComm->newContextBackward(iteration - 1, gradTensor, outputTensor, savedTensors, localVerticesLabels, vtcsCnt, inFeatDim, outFeatDim, getFeatDim(numLayers));
    resComm->requestBackward(iteration - 1, iteration - 1 == numLayers - 1);
    // printLog(nodeId, "All lambda requests finished. Results received.");

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

FeatType *
Engine::scatterBackward(FeatType *gradTensor, unsigned vtcsCnt, unsigned featDim) {
    double sttTimer = getTimer();

    // Start data communicators.
    commHalt = false;
    backwardGhostVerticesData = new FeatType [featDim * graph.getNumOutEdgeGhostVertices()];
    auto fgc_fp = std::bind(&Engine::backwardGhostCommunicator, this, std::placeholders::_1, std::placeholders::_2);
    dataPool->perform(fgc_fp);

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
    vecTimeSendout[numLayers + iteration - 1] += getTimer() - sttTimer;
    return gradTensor;
}

void
Engine::aggregateBPCompute(unsigned tid, void *args) {
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
    Vertex &v = graph.getVertex(lvid);
    for (unsigned i = 0; i < featDim; ++i) {
        currDataDst[i] *= v.getNormFactor();
    }

    // Aggregate from incoming neighbors.
    for (unsigned i = 0; i < v.getNumOutEdges(); ++i) {
        FeatType *otherDataPtr;
        EdgeType normFactor = v.getOutEdge(i).getData();

        if (v.getOutEdge(i).getEdgeLocation() == LOCAL_EDGE_TYPE) {    // Local vertex.
            otherDataPtr = getVtxFeat(gradTensor, v.getDestVertexLocalId(i), featDim);
        } else {                                                       // Ghost vertex.
            otherDataPtr = getVtxFeat(backwardGhostVerticesData, v.getDestVertexLocalId(i), featDim);
        }
        // TODO: Locks on the data array area is not properly set yet. But does not affect forward prop.
        for (unsigned j = 0; j < featDim; ++j) {
            currDataDst[j] += otherDataPtr[j] * normFactor;
        }
    }
}

void
Engine::sendBackwardGhostGradients(FeatType *gradTensor, unsigned featDim) {
    // Loop through all local vertices and do the data send out work. If there are any remote edges for a vertex, should send this vid to
    // other nodes for their ghost's update.
    // TODO: (YIFAN) process crashes when return if BATCH_SIZE is too large. Weird, to be fixed.
    // Please decrease BATCH_SIZE if porcess crashed.
    bool batchFlag = false;
    unsigned BATCH_SIZE = std::max(((batchFlag ? MAX_MSG_SIZE : 4096) - DATA_HEADER_SIZE) /
                                   (sizeof(unsigned) + sizeof(FeatType) * featDim), 1ul); // at least send one vertex
    for (unsigned nid = 0; nid < numNodes; ++nid) {
        if (nid == nodeId) {
            continue;
        }
        unsigned backwardGhostVCnt = backwardGhostsList[nid].size();
        for (unsigned ib = 0; ib < backwardGhostVCnt; ib += BATCH_SIZE) {
            unsigned sendBatchSize = (backwardGhostVCnt - ib) < BATCH_SIZE ? (backwardGhostVCnt - ib) : BATCH_SIZE;

            backwardVerticesPushOut(nid, sendBatchSize, backwardGhostsList[nid].data() + ib, gradTensor, featDim);
            lockRecvCnt.lock();
            recvCnt++;
            lockRecvCnt.unlock();
        }
    }
    // Wait for all remote schedulings sent by me to be handled.
    lockRecvCnt.lock();
    if (recvCnt > 0) {
        condRecvCnt.wait();
    }
    lockRecvCnt.unlock();
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
 * Major part of the engine's communication logic is done by data threads. These threads loop asynchronously with computation workers.
 *
 */
void
Engine::backwardGhostCommunicator(unsigned tid, void *args) {
    // backoff sleep strategy to improve CPU utilization
    int failedTrials = 0;
    const int INIT_PERIOD = 256;
    const int MAX_PERIOD = 4096;
    int SLEEP_PERIOD = INIT_PERIOD;
    unsigned sender, topic;
    FeatType *msgBuf = (FeatType *)new char[MAX_MSG_SIZE];

    // While loop, looping infinitely to get the next message.
    while (!commHalt) {
        // No message in queue.
        if (!commManager.dataPullIn(&sender, &topic, msgBuf, MAX_MSG_SIZE)) {
            // Computation workers done their work, so communicator goes to death as well.
            if (commHalt) {
                break;
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
                unsigned featDim = getFeatDim(iteration - 1);
                // Update ghost vertices
                for (unsigned i = 0; i < recvGhostVCnt; ++i) {
                    unsigned gvid = *(unsigned *)bufPtr;
                    bufPtr += sizeof(unsigned);
                    FeatType *dataPtr = getVtxFeat(backwardGhostVerticesData, graph.getOutEdgeGhostVertex(gvid).getLocalId(), featDim);
                    memcpy(dataPtr, bufPtr, sizeof(FeatType) * featDim);
                    bufPtr += sizeof(FeatType) * featDim;
                }

                // A respond to a broadcast, and the topic vertex is in my local vertices. I should update the
                // corresponding recvWaiter's value. If waiters become empty, send a signal in case the workers are
                // waiting on it to be empty at the iteration barrier.
            } else { // (topic == MAX_IDTYPE - 1)
                lockRecvCnt.lock();
                recvCnt--;
                if (recvCnt == 0) {
                    condRecvCnt.signal();
                }
                lockRecvCnt.unlock();
            }

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
}


/**
 *
 * Print engine metrics of processing time.
 *
 */
void
Engine::printEngineMetrics() {
    printLog(nodeId, "<EM>: Initialization takes %.3lf ms", timeInit);
    printLog(nodeId, "<EM>: Forward:  Time per stage:");
    for (unsigned i = 0; i < numLayers; ++i) {
        printLog(nodeId, "<EM>    Aggregation   %2u  %.3lf ms", i, vecTimeAggregate[i]);
        printLog(nodeId, "<EM>    Lambda        %2u  %.3lf ms", i, vecTimeLambda[i]);
        printLog(nodeId, "<EM>    Ghost update  %2u  %.3lf ms", i, vecTimeSendout[i]);
    }
    printLog(nodeId, "<EM>: Total forward-prop time %.3lf ms", timeForwardProcess);

    printLog(nodeId, "<EM>: Backward: Time per stage:");
    for (unsigned i = numLayers; i < 2 * numLayers; i++) {
        printLog(nodeId, "<EM>    Aggregation   %2u  %.3lf ms", i, vecTimeAggregate[i]);
        printLog(nodeId, "<EM>    Lambda        %2u  %.3lf ms", i, vecTimeLambda[i]);
        printLog(nodeId, "<EM>    Ghost update  %2u  %.3lf ms", i, vecTimeSendout[i]);
    }
    printLog(nodeId, "<EM>: Backward-prop takes %.3lf ms", timeBackwardProcess);
}


/**
 *
 * Print my graph's metrics.
 *
 */
void
Engine::printGraphMetrics() {
    printLog(nodeId, "<GM>: %u global vertices, %llu global edges, %u local vertices.",
             graph.getNumGlobalVertices(), graph.getNumGlobalEdges(), graph.getNumLocalVertices());
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

    ("graphfile", boost::program_options::value<std::string>(), "Path to the binary file contatining the edge list")
    ("featuresfile", boost::program_options::value<std::string>(), "Path to the file containing the vertex features")
    ("layerfile", boost::program_options::value<std::string>(), "Layer configuration file")
    ("labelsfile", boost::program_options::value<std::string>(), "Target labels file")
    ("dshmachinesfile", boost::program_options::value<std::string>(), "DSH machines file")
    ("pripfile", boost::program_options::value<std::string>(), "File containing my private ip")
    ("pubipfile", boost::program_options::value<std::string>(), "File containing my public ip")

    ("tmpdir", boost::program_options::value<std::string>(), "Temporary directory")

    ("dataserverport", boost::program_options::value<unsigned>(), "The port exposing to the coordination server")
    ("weightserverport", boost::program_options::value<unsigned>(), "The port of the listener on the coordination server")
    ("wserveripfile", boost::program_options::value<std::string>(), "The file contains the public IP addresses of the weight server")
    ("coordserverip", boost::program_options::value<std::string>(), "The private IP address of the coordination server")
    ("coordserverport", boost::program_options::value<unsigned>(), "The port of the listener on the coordination server")

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

    ("GPU", boost::program_options::value<unsigned>(), "Enable GPU or not")
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

    assert(vm.count("graphfile"));
    graphFile = vm["graphfile"].as<std::string>();

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

    assert(vm.count("coordserverip"));
    coordserverIp = vm["coordserverip"].as<std::string>();

    assert(vm.count("coordserverport"));
    coordserverPort = vm["coordserverport"].as<unsigned>();

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
    //    numEpochs = vm["numEpochs"].as<unsigned>();

    assert(vm.count("validationFrequency"));
    //    valFreq = vm["validationFrequency"].as<unsigned>();

    assert(vm.count("GPU"));
    gpuEnabled = vm["GPU"].as<unsigned>();

    printLog(404, "Parsed configuration: dThreads = %u, cThreads = %u, graphFile = %s, featuresFile = %s, dshMachinesFile = %s, "
             "myPrIpFile = %s, myPubIpFile = %s, undirected = %s, data port set -> %u, control port set -> %u, node port set -> %u",
             dThreads, cThreads, graphFile.c_str(), featuresFile.c_str(), dshMachinesFile.c_str(),
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
        if (graph.containsInEdgeGhostVertex(gvid)) {      // Global vertex.
            FeatType *actDataPtr = getVtxFeat(forwardGhostInitData, graph.getInEdgeGhostVertex(gvid).getLocalId(), featDim);
            memcpy(actDataPtr, feature_vec.data(), featDim * sizeof(FeatType));
        } else if (graph.containsVertex(gvid)) {    // Local vertex.
            FeatType *actDataPtr = getVtxFeat(forwardVerticesInitData, graph.getVertexByGlobal(gvid).getLocalId(), featDim);
            memcpy(actDataPtr, feature_vec.data(), featDim * sizeof(FeatType));
        }
        ++gvid;
    }
    infile.close();
    assert(gvid == graph.getNumGlobalVertices());
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
        if (graph.containsVertex(gvid)) {

            // Convert into a one-hot array.
            assert(curr < lKinds);
            memset(one_hot_arr, 0, lKinds * sizeof(FeatType));
            one_hot_arr[curr] = 1.0;

            FeatType *labelPtr = localVertexLabelsPtr(graph.getVertexByGlobal(gvid).getLocalId());
            memcpy(labelPtr, one_hot_arr, lKinds * sizeof(FeatType));
        }

        ++gvid;
    }

    infile.close();
    assert(gvid == graph.getNumGlobalVertices());
}


/**
 *
 * Read in the partition file.
 *
 */
void
Engine::readPartsFile(std::string &partsFileName, Graph &lGraph) {
    std::ifstream infile(partsFileName.c_str());
    if (!infile.good())
        printLog(nodeId, "Cannot open patition file: %s [Reason: %s]", partsFileName.c_str(), std::strerror(errno));

    assert(infile.good());

    short partId;
    unsigned lvid = 0;
    unsigned gvid = 0;

    std::string line;
    while (std::getline(infile, line)) {
        if (line.size() == 0 || (line[0] < '0' || line[0] > '9'))
            continue;

        std::istringstream iss(line);
        if (!(iss >> partId))
            break;

        lGraph.appendVertexPartitionId(partId);

        if (partId == nodeId) {
            lGraph.localToGlobalId[lvid] = gvid;
            lGraph.globalToLocalId[gvid] = lvid;

            ++lvid;
        }
        ++gvid;
    }

    lGraph.setNumGlobalVertices(gvid);
    lGraph.setNumLocalVertices(lvid);
}


/**
 *
 * Process an edge read from the binary snap file.
 *
 */
void
Engine::processEdge(unsigned &from, unsigned &to, Graph &lGraph, bool **forwardGhostVTables, bool **backwardGhostVTables) {
    if (lGraph.getVertexPartitionId(from) == nodeId) {
        unsigned lFromId = lGraph.globalToLocalId[from];
        unsigned toId;
        EdgeLocationType eLocation;

        unsigned toPartition = lGraph.getVertexPartitionId(to);
        if (toPartition == nodeId) {
            toId = lGraph.globalToLocalId[to];
            eLocation = LOCAL_EDGE_TYPE;
        } else {
            toId = to;
            eLocation = REMOTE_EDGE_TYPE;
            lGraph.getVertex(lFromId).setVertexLocation(BOUNDARY_VERTEX);

            if (!lGraph.containsOutEdgeGhostVertex(to)) {
                lGraph.getOutEdgeGhostVertices()[to] = GhostVertex();
            }
            lGraph.getOutEdgeGhostVertex(to).addAssocEdge(lFromId);

            forwardGhostVTables[toPartition][lFromId] = true;
        }

        lGraph.getVertex(lFromId).addOutEdge(OutEdge(toId, eLocation, EdgeType()));
    }

    if (lGraph.getVertexPartitionId(to) == nodeId) {
        unsigned lToId = lGraph.globalToLocalId[to];
        unsigned fromId;
        EdgeLocationType eLocation;

        unsigned fromPartition = lGraph.getVertexPartitionId(from);
        if (fromPartition == nodeId) {
            fromId = lGraph.globalToLocalId[from];
            eLocation = LOCAL_EDGE_TYPE;
        } else {
            fromId = from;
            eLocation = REMOTE_EDGE_TYPE;

            if (!lGraph.containsInEdgeGhostVertex(from)) {
                lGraph.getInEdgeGhostVertices()[from] = GhostVertex();
            }
            lGraph.getInEdgeGhostVertex(from).addAssocEdge(lToId);

            backwardGhostVTables[fromPartition][lToId] = true;
        }

        lGraph.getVertex(lToId).addInEdge(InEdge(fromId, eLocation, EdgeType()));
    }
}


/**
 *
 * Set the normalization factors on all edges.
 *
 */
void
Engine::setEdgeNormalizations() {
    for (Vertex &vertex : graph.getVertices()) {
        unsigned vtxDeg = vertex.getNumInEdges() + 1;
        float vtxNorm = std::pow(vtxDeg, -.5);
        vertex.setNormFactor(vtxNorm * vtxNorm);
        for (unsigned i = 0; i < vertex.getNumInEdges(); ++i) {
            InEdge &e = vertex.getInEdge(i);
            unsigned vid = e.getSourceId();
            if (e.getEdgeLocation() == LOCAL_EDGE_TYPE) {
                unsigned srcDeg = graph.getVertex(vid).getNumInEdges() + 1;
                float srcNorm = std::pow(srcDeg, -.5);
                e.setData(srcNorm * vtxNorm);
            } else {
                unsigned ghostDeg = graph.getInEdgeGhostVertex(vid).getDegree() + 1;
                float ghostNorm = std::pow(ghostDeg, -.5);
                e.setData(ghostNorm * vtxNorm);
            }
        }
        for (unsigned i = 0; i < vertex.getNumOutEdges(); ++i) {
            OutEdge &e = vertex.getOutEdge(i);
            unsigned vid = e.getDestId();
            if (e.getEdgeLocation() == LOCAL_EDGE_TYPE) {
                unsigned dstDeg = graph.getVertex(vid).getNumInEdges() + 1;
                float dstNorm = std::pow(dstDeg, -.5);
                e.setData(vtxNorm * dstNorm);
            } else {
                unsigned ghostDeg = graph.getOutEdgeGhostVertex(vid).getDegree() + 1;
                float ghostNorm = std::pow(ghostDeg, -.5);
                e.setData(vtxNorm * ghostNorm);
            }
        }
    }
}


/**
 *
 * Finds the in degree of all ghost vertices.
 *
 */
void
Engine::findGhostDegrees(std::string &fileName) {
    std::ifstream infile(fileName.c_str(), std::ios::binary);
    if (!infile.good())
        printLog(nodeId, "Cannot open BinarySnap file: %s", fileName.c_str());

    assert(infile.good());

    BSHeaderType bsHeader;
    infile.read((char *) &bsHeader, sizeof(bsHeader));

    unsigned srcdst[2];
    while (infile.read((char *) srcdst, bsHeader.sizeOfVertexType * 2)) {
        if (srcdst[0] == srcdst[1]) {
            continue;
        }

        // YIFAN: we count in degree for both outEdgeGhosts and inEdgeGhosts
        if (graph.containsOutEdgeGhostVertex(srcdst[1])) {
            graph.getOutEdgeGhostVertex(srcdst[1]).incrementDegree();
        }
        if (graph.containsInEdgeGhostVertex(srcdst[1])) {
            graph.getInEdgeGhostVertex(srcdst[1]).incrementDegree();
        }
    }

    infile.close();
}


/**
 *
 * Read and parse the graph from the graph binary snap file.
 *
 */
void
Engine::readGraphBS(std::string &fileName) {

    // Read in the partition file.
    std::string partsFileName = fileName + PARTS_EXT;
    readPartsFile(partsFileName, graph);

    // Initialize the graph based on the partition info.
    graph.getVertices().resize(graph.getNumLocalVertices());
    for (unsigned i = 0; i < graph.getNumLocalVertices(); ++i) {
        graph.getVertex(i).setLocalId(i);
        graph.getVertex(i).setGlobalId(graph.localToGlobalId[i]);
        graph.getVertex(i).setVertexLocation(INTERNAL_VERTEX);
        graph.getVertex(i).setGraphPtr(&graph);
    }

    // Read in the binary snap edge file.
    std::string edgeFileName = fileName + EDGES_EXT;
    std::ifstream infile(edgeFileName.c_str(), std::ios::binary);
    if(!infile.good())
        printLog(nodeId, "Cannot open BinarySnap file: %s", edgeFileName.c_str());

    assert(infile.good());

    BSHeaderType bSHeader;
    infile.read((char *) &bSHeader, sizeof(bSHeader));
    assert(bSHeader.sizeOfVertexType == sizeof(unsigned));

    bool **forwardGhostVTables = new bool* [numNodes];
    bool **backwardGhostVTables = new bool* [numNodes];
    for (unsigned i = 0; i < numNodes; ++i) {
        if (i == nodeId) {
            continue;
        }
        forwardGhostVTables[i] = new bool [graph.getNumLocalVertices()];
        memset(forwardGhostVTables[i], 0, sizeof(bool) * graph.getNumLocalVertices());
        backwardGhostVTables[i] = new bool [graph.getNumLocalVertices()];
        memset(backwardGhostVTables[i], 0, sizeof(bool) * graph.getNumLocalVertices());
    }

    // Loop through all edges and process them.
    unsigned srcdst[2];
    while (infile.read((char *) srcdst, bSHeader.sizeOfVertexType * 2)) {
        if (srcdst[0] == srcdst[1])
            continue;

        processEdge(srcdst[0], srcdst[1], graph, forwardGhostVTables, backwardGhostVTables);
        if (undirected)
            processEdge(srcdst[1], srcdst[0], graph, forwardGhostVTables, backwardGhostVTables);
        graph.incrementNumGlobalEdges();
    }

    for (unsigned i = 0; i < numNodes; ++i) {
        if (i == nodeId) {
            continue;
        }
        forwardGhostsList[i].reserve(graph.getNumLocalVertices());
        backwardGhostsList[i].reserve(graph.getNumLocalVertices());
        for (unsigned j = 0; j < graph.getNumLocalVertices(); ++j) {
            if (forwardGhostVTables[i][j]) {
                forwardGhostsList[i].push_back(j);
            }
            if (backwardGhostVTables[i][j]) {
                backwardGhostsList[i].push_back(j);
            }
        }
        forwardGhostsList[i].shrink_to_fit();
        backwardGhostsList[i].shrink_to_fit();
        delete[] forwardGhostVTables[i];
        delete[] backwardGhostVTables[i];
    }
    delete[] forwardGhostVTables;
    delete[] backwardGhostVTables;

    infile.close();

    // Extra works added.
    graph.setNumInEdgeGhostVertices(graph.getInEdgeGhostVertices().size());
    graph.setNumOutEdgeGhostVertices(graph.getOutEdgeGhostVertices().size());
    findGhostDegrees(edgeFileName);
    setEdgeNormalizations();
}

// Substantiate an Engine object here for TF aggregators
Engine engine;
