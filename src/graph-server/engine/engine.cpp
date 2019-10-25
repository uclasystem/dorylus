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
    forwardGhostVCnts = new unsigned [numNodes];
    backwardGhostVCnts = new unsigned [numNodes];
    memset(forwardGhostVCnts, 0, sizeof(unsigned) * numNodes);
    memset(backwardGhostVCnts, 0 ,sizeof(unsigned) * numNodes);
    forwardBatchMsgBuf = new unsigned *[numNodes];
    backwardBatchMsgBuf = new unsigned *[numNodes];
    // Read in the graph and subscribe vertex global ID topics.
    std::set<unsigned> inTopics;
    std::vector<unsigned> outTopics;
    readGraphBS(graphFile, inTopics, outTopics);
    printGraphMetrics();

    // Create the global contiguous memory for vertices' data, according to the given layer config and number of local vertices.
    // Create the global contiguous memory for ghost vertices' data similarly. Then read in initial features.
    localVerticesZData = new FeatType *[numLayers + 1];
    localVerticesActivationData = new FeatType *[numLayers + 1];
    savedTensors = new std::vector<Matrix> [numLayers];

    // Init it here for collecting data when reading files
    forwardGhostInitData = new FeatType[layerConfig[0] * graph.getNumInEdgeGhostVertices()];
    // Create data storage area for each layer.
    localVerticesZData[0] = new FeatType [layerConfig[0] * graph.getNumLocalVertices()];
    localVerticesActivationData[0] = new FeatType [layerConfig[0] * graph.getNumLocalVertices()];
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
    lockCurrId.init();
    recvCnt = 0;
    lockRecvCnt.init();
    condRecvCnt.init(lockRecvCnt);
    lockHalt.init();

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

    if (gpuEnabled == 1)
        resComm=createResourceComm("GPU",commInfo);
    else
        resComm=createResourceComm("Lambda",commInfo);

    timeForwardProcess = 0.0;
    timeInit += getTimer();
    printLog(nodeId, "Engine initialization complete.");
}


/**
*
* Initialize the CommInfo struct for lambdaComm and GPUComm
*
*/
void
Engine::setUpCommInfo(){
    commInfo.nodeIp=nodeManager.getNode(nodeId).pubip;
    commInfo.nodeId=nodeId;
    commInfo.dataserverPort=dataserverPort;
    commInfo.coordserverIp=coordserverIp;
    commInfo.coordserverPort=coordserverPort;
    commInfo.numLambdasForward=numLambdasForward;
    commInfo.numLambdasBackward=numLambdasBackward;
    commInfo.numNodes=numNodes;
    commInfo.wServersFile=weightserverIPFile;
    commInfo.weightserverPort=weightserverPort;
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

inline size_t argmax(FeatType* first, FeatType* last) {
    return std::distance(first, std::max_element(first, last));
}
inline void calcAcc(Engine *e, FeatType *predicts, FeatType *labels, unsigned vtcsCnt, unsigned featDim) {
    float acc = 0.0;
    float loss = 0.0;
    for (unsigned i = 0; i < vtcsCnt; i++) {
        FeatType *currLabel = labels + i * featDim;
        FeatType *currPred = predicts + i * featDim;
        acc += currLabel[argmax(currPred, currPred + featDim)];
        for (unsigned j = 0; j < featDim; j++) {
            loss -= std::log(currPred[argmax(currLabel, currLabel + featDim)] + 1e-20);
        }
    }
    acc /= vtcsCnt;
    loss /= vtcsCnt;
    printLog(e->getNodeId(), "batch loss %f, batch acc %f", loss, acc);
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
    halt = false;
    commHalt = false;

    const unsigned graphLocalVerticesNum = graph.getNumLocalVertices();
    // Create buffer for first-layer aggregation.
    FeatType *inputTensor = localVerticesActivationData[0];
    forwardGhostVerticesData = forwardGhostInitData;
    for (iteration = 0; iteration < numLayers; iteration++) {
        inputTensor = aggregate(inputTensor, graphLocalVerticesNum, getFeatDim(iteration));
        inputTensor = invokeLambda(inputTensor, graphLocalVerticesNum, getFeatDim(iteration), getFeatDim(iteration + 1));
        inputTensor = scatter(inputTensor, graphLocalVerticesNum, getFeatDim(iteration + 1));
    }

    timeForwardProcess += getTimer();
    printLog(nodeId, "Engine completes FORWARD at iter %u.", iteration);
    calcAcc(this, inputTensor, localVerticesLabels, graphLocalVerticesNum, getFeatDim(numLayers));

    return inputTensor;
}


/**
 *
 * Runs a backward propagation phase: (Lambda Computing) -> ( ... ) -> ...
 * Will start a bunch of worker threads and a bunch of data communicator threads.
 *
 */
void
Engine::runBackward(FeatType *backwardInitData) {
    // Make sure all nodes start running the backward-prop phase.
    nodeManager.barrier();
    printLog(nodeId, "Engine starts running BACKWARD...");

    timeBackwardProcess = getTimer();

    // Set initial conditions.
    currId = 0;
    iteration = numLayers;
    halt = false;
    commHalt = false;

    bool disableBackward = false;
    if (disableBackward) {
        timeBackwardProcess = getTimer() - timeBackwardProcess;
        // We have to set backward to tell the location of labels matrix to communicator for forward evaluation.
        // TOD): (YIFAN) set localVerticesLabels in forward context so that we won't need this hack.
        resComm->newContextBackward(localVerticesZData, localVerticesActivationData, localVerticesLabels,
                                            graph.getNumLocalVertices(), layerConfig);
        for (unsigned i = 0; i < numLayers; ++i) {
            vecTimeAggregate.push_back(0.0);
            vecTimeLambda.push_back(0.0);
            vecTimeSendout.push_back(0.0);
        }
        printLog(nodeId, "Engine skips BACKWARD propagation.");
    } else {
        // Start data communicators.
        auto bgc_fp = std::bind(&Engine::backwardGhostCommunicator, this, std::placeholders::_1, std::placeholders::_2);
        dataPool->perform(bgc_fp);

        // Start backward-prop workers.
        auto bw_fp = std::bind(&Engine::backwardWorker, this, std::placeholders::_1, std::placeholders::_2);
        computePool->perform(bw_fp, backwardInitData);

        // Join all backward-prop workers.
        computePool->sync();

        timeBackwardProcess = getTimer() - timeBackwardProcess;

        // Join all data communicators.
        dataPool->sync();
    }

    // skip index 0. delete buffer[0] is inside destory()
    for (size_t i = 1; i <= numLayers; ++i) {
        delete[] localVerticesZData[i];
        delete[] localVerticesActivationData[i];
        // for (size_t j = 0; j < savedTensors[i-1].size(); j++) {
        //     delete[] savedTensors[i-1][j].getData();
        // }
        savedTensors[i-1].clear();
    }

    printLog(nodeId, "Engine completes BACKWARD propagation.");
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
    // The following are full feature values outputing.
    //
    // for (Vertex& v : graph.getVertices()) {
    //     outStream << v.getGlobalId() << ": ";
    //     for (size_t i = 0; i <= numLayers; ++i) {
    //         FeatType *dataPtr = getVtxFeat(localVerticesActivationData[i], v.getLocalId(), getFeatDim(i));
    //         for (size_t j = 0; j < layerConfig[i]; ++j)
    //             outStream << dataPtr[j] << " ";
    //         outStream << "| ";
    //     }
    //     outStream << std::endl;
    // }

    //
    // The follwing are only-last-layer feature values outputing.
    //
    for (Vertex& v : graph.getVertices()) {
        outStream << v.getGlobalId() << ": ";
        FeatType *dataPtr = getVtxFeat(localVerticesActivationData[numLayers], v.getLocalId(), getFeatDim(numLayers));
        for (size_t j = 0; j < layerConfig[numLayers]; ++j)
            outStream << dataPtr[j] << " ";
        outStream << std::endl;
    }

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

    // Write benchmarking results to log file.
    if (master()) {
        assert(vecTimeAggregate.size() == numLayers);
        assert(vecTimeLambda.size() == numLayers);
        assert(vecTimeSendout.size() == numLayers);
        printEngineMetrics();
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

    lockCurrId.destroy();
    lockRecvCnt.destroy();
    condRecvCnt.destroy();
    lockHalt.destroy();

    resComm->sendShutdownMessage();

    delete[] forwardGhostVCnts;
    delete[] backwardGhostVCnts;
    for (size_t i = 0; i < numNodes; ++i) {
        if (i != nodeId) {
            delete[] forwardBatchMsgBuf[i];
            delete[] backwardBatchMsgBuf[i];
        }
    }
    delete[] forwardBatchMsgBuf;
    delete[] backwardBatchMsgBuf;

    // corresponding to the new [] in init()
    delete[] localVerticesZData[0];
    delete[] localVerticesActivationData[0];

    delete[] localVerticesZData;
    delete[] localVerticesActivationData;
    delete[] forwardGhostInitData;
    delete[] forwardGhostVerticesData;

    bool disableBackward = false;
    if (!disableBackward) {
        delete[] backwardGhostVerticesData;
    }

    // delete[] localVerticesDataBuf;
    delete[] localVerticesLabels;
}

struct AggOPArgs {
    FeatType *outputTensor;
    FeatType *vtcsTensor;
    unsigned vtcsCnt;
    unsigned featDim;
};

FeatType* Engine::aggregate(FeatType *vtcsTensor, unsigned vtcsCnt, unsigned featDim) {
    double sttTimer = getTimer();

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

    return outputTensor;
}

FeatType *
Engine::invokeLambda(FeatType *vtcsTensor, unsigned vtcsCnt, unsigned inFeatDim, unsigned outFeatDim) {
    double sttTimer = getTimer();

    assert(vtcsCnt == graph.getNumLocalVertices());

    // TODO: (YIFAN) actually we don't need to new a saved tensor since we can reuse the input. Optimize the lambda later to send only a flag back.
    FeatType *savedTensor; // = new FeatType [vtcsCnt * inFeatDim];
    // TODO: (YIFAN) Currently we cannot make sure to prevent outputTensor to be deleted. So new it first.
    localVerticesZData[iteration + 1] = new FeatType [vtcsCnt * outFeatDim]; // TODO: (YIFAN) useless. change this!
    FeatType *outputTensor = new FeatType [vtcsCnt * outFeatDim];

    bool runEval = evaluate && iteration == numLayers-1;
    // Start a new lambda communication context.
    resComm->newContextForward(vtcsTensor, localVerticesZData[iteration + 1], outputTensor, vtcsCnt, inFeatDim, outFeatDim, runEval);

    // if in GPU mode we launch gpu computation here and wait the results
    if (gpuEnabled) {
        resComm->requestForward(iteration);
    } else {
        unsigned availLambdaId = 0;
        while(availLambdaId < numLambdasForward) {
            resComm->invokeLambdaForward(iteration, availLambdaId, iteration == numLayers - 1);
            availLambdaId++;
        }
        resComm->waitLambdaForward();
    }

    bool saveInput = true;
    // Reuse inputTensor if vertex NN wants to save it for backward computation.
    if (saveInput) {
        savedTensor = vtcsTensor;
        vtcsTensor = NULL;
        savedTensors[iteration].push_back(Matrix(graph.getNumLocalVertices(), inFeatDim, savedTensor));
    } else {
        delete[] vtcsTensor;
    }
    savedTensors[iteration].push_back(Matrix(graph.getNumLocalVertices(), outFeatDim, localVerticesZData[iteration + 1]));
    bool saveOutput = true;
    if (saveOutput) {
        // Currently we cannot make sure to prevent outputTensor to be deleted. So we copy it first.
        // TODO: (YIFAN) thinking about if we can optimize this.
        // localVerticesActivationData[iteration + 1] = outputTensor;
        localVerticesActivationData[iteration + 1] = new FeatType [vtcsCnt * outFeatDim];
        memcpy(localVerticesActivationData[iteration + 1], outputTensor, vtcsCnt * outFeatDim * sizeof(FeatType));
        savedTensors[iteration].push_back(Matrix(graph.getNumLocalVertices(), outFeatDim, localVerticesActivationData[iteration + 1]));
    }

    if (vecTimeLambda.size() < numLayers) {
        vecTimeLambda.push_back(getTimer() - sttTimer);
    } else {
        vecTimeLambda[iteration] += getTimer() - sttTimer;
    }
    printLog(nodeId, "All lambda requests finished. Results received.");

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

    //## Global Iteration barrier. ##//
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
    FeatType *vtcsTensor = ((AggOPArgs *) args)->vtcsTensor;
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
    Vertex& v = graph.getVertex(lvid);
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
    bool batchFlag = false;
    unsigned BATCH_SIZE = std::max(((batchFlag ? MAX_MSG_SIZE : 4096) - DATA_HEADER_SIZE) /
                        (sizeof(unsigned) + sizeof(FeatType) * featDim), 1ul); // at least send one vertex
    for (unsigned nid = 0; nid < numNodes; ++nid) {
        if (nid == nodeId) {
            continue;
        }
        unsigned forwardGhostVCnt = forwardGhostVCnts[nid];
        for (unsigned ib = 0; ib < forwardGhostVCnt; ib += BATCH_SIZE) {
            unsigned sendBatchSize = (forwardGhostVCnt - ib) < BATCH_SIZE ? (forwardGhostVCnt - ib) : BATCH_SIZE;

            forwardVerticesPushOut(nid, sendBatchSize, forwardBatchMsgBuf[nid] + ib, inputTensor, featDim);
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
void
Engine::backwardWorker(unsigned tid, void *args) {
    // A Stopwatch for composing steps of run().
    double timeWorker = getTimer();

    FeatType *backwardInitData = (FeatType *) args;

    const unsigned graphLocalVerticesNum = graph.getNumLocalVertices();
    while (iteration > 0) {
        if (tid == 0) {
            printLog(nodeId, "Iteration %u starts.", iteration);
            if (iteration == numLayers) {
                aggGradData = backwardInitData;
            } else if (iteration < numLayers) {
                delete[] backwardGhostVerticesData;
            }
            backwardGhostVerticesData = new FeatType[layerConfig[iteration - 1] * graph.getNumOutEdgeGhostVertices()];
            newGradData = new FeatType[layerConfig[iteration - 1] * graphLocalVerticesNum];

            // Start a new lambda communication context.
            // TODO: (YIFAN) currently set new context for backward once is enough. Change the newContextBackward interface
            resComm->newContextBackward(aggGradData, newGradData, savedTensors, localVerticesLabels, graphLocalVerticesNum, getFeatDim(iteration - 1), getFeatDim(iteration), getFeatDim(numLayers));
            resComm->requestBackward(iteration - 1, iteration - 1 == numLayers - 1);
            vecTimeLambda.push_back(getTimer() - timeWorker);
            timeWorker = getTimer();
            printLog(nodeId, "All lambda requests finished. Results received.");

            delete[] aggGradData;
            aggGradData = new FeatType[layerConfig[iteration - 1] * graphLocalVerticesNum];

            sendBackwardGhostGradients();

            vecTimeSendout.push_back(getTimer() - timeWorker);

            //## Global Iteration barrier. ##//
            nodeManager.barrier();
            timeWorker = getTimer();
            printLog(nodeId, "Global barrier after ghost data exchange crossed.");

            currId = 0;
        }
        barComp.wait();

        unsigned lvid = 0;
        while (currId < graphLocalVerticesNum) {
            lockCurrId.lock();
            lvid = currId++;
            lockCurrId.unlock();

            if (lvid < graphLocalVerticesNum) {
                // Backward Aggregation.
                backwardAggregateFromNeighbors(lvid);
            }
        }
        if (tid == 0) {
            delete[] newGradData;
            vecTimeAggregate.push_back(getTimer() - timeWorker);
            timeWorker = getTimer();

            iteration--;
        }
        barComp.wait();
    }
    if (tid == 0) {
        // delete[] aggGradData;
        // delete[] backwardGhostVerticesData;
        printLog(nodeId, "Deciding to halt at iteration %u...", 0);
        lockHalt.lock();
        halt = true;
        commHalt = true;
        lockHalt.unlock();
    }
}


/**
 *
 * Aggregate featDim feature values starting from offset from all neighbors (including self). Then write the results to the
 * data buffer area for serialization. The results are to be used for being sent to lambda threads.
 *
 */
void
Engine::backwardAggregateFromNeighbors(unsigned lvid) {
    unsigned featDim = getFeatDim(iteration - 1);

    // Read out data of the current iteration of given vertex.
    FeatType *currDataDst = getVtxFeat(aggGradData, lvid, featDim);
    FeatType *currDataPtr = getVtxFeat(newGradData, lvid, featDim);

    memcpy(currDataDst, currDataPtr, featDim * sizeof(FeatType));

    // Apply normalization factor on the current data.
    Vertex& v = graph.getVertex(lvid);
    for (unsigned i = 0; i < featDim; ++i) {
        currDataDst[i] *= v.getNormFactor();
    }

    // Aggregate from incoming neighbors.
    for (unsigned i = 0; i < v.getNumOutEdges(); ++i) {
        FeatType *otherDataPtr;
        EdgeType normFactor = v.getOutEdge(i).getData();

        if (v.getOutEdge(i).getEdgeLocation() == LOCAL_EDGE_TYPE) {    // Local vertex.
            otherDataPtr = getVtxFeat(newGradData, v.getDestVertexLocalId(i), featDim);
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
Engine::sendBackwardGhostGradients() {
    // Loop through all local vertices and do the data send out work. If there are any remote edges for a vertex, should send this vid to
    // other nodes for their ghost's update.
    // TODO: (YIFAN) process crashes when return if BATCH_SIZE is too large. Weird, to be fixed.
    // Please decrease BATCH_SIZE if porcess crashed.
    bool batchFlag = false;
    unsigned BATCH_SIZE = std::max(((batchFlag ? MAX_MSG_SIZE : 4096) - DATA_HEADER_SIZE) /
                        (sizeof(unsigned) + sizeof(FeatType) * getFeatDim(iteration - 1)), 1ul); // at least send one vertex
    for (unsigned nid = 0; nid < numNodes; ++nid) {
        if (nid == nodeId) {
            continue;
        }
        unsigned recvGhostVCnt = backwardGhostVCnts[nid];
        for (unsigned ib = 0; ib < recvGhostVCnt; ib += BATCH_SIZE) {
            unsigned sendBatchSize = (recvGhostVCnt - ib) < BATCH_SIZE ? (recvGhostVCnt - ib) : BATCH_SIZE;

            backwardVerticesPushOut(nid, nodeId, sendBatchSize, backwardBatchMsgBuf[nid] + ib);
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
Engine::backwardVerticesPushOut(unsigned receiver, unsigned sender, unsigned totCnt, unsigned *lvids) {
    const unsigned featDim = getFeatDim(iteration - 1);
    zmq::message_t msg(DATA_HEADER_SIZE + (sizeof(unsigned) + sizeof(FeatType) * featDim) * totCnt);
    char *msgPtr = (char *)(msg.data());
    sprintf(msgPtr, NODE_ID_HEADER, receiver);
    msgPtr += NODE_ID_DIGITS;
    *(unsigned *)msgPtr = sender;
    msgPtr += sizeof(unsigned);
    *(unsigned *)msgPtr = totCnt;
    msgPtr += sizeof(unsigned);
    for (unsigned i = 0; i < totCnt; ++i) {
        *(unsigned *)msgPtr = graph.localToGlobalId[lvids[i]];
        msgPtr += sizeof(unsigned);
        FeatType *dataPtr = getVtxFeat(newGradData, lvids[i], featDim);
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
 * Print engine metrics of processing time.
 *
 */
void
Engine::printEngineMetrics() {
    printLog(nodeId, "<EM>: Initialization takes %.3lf ms", timeInit);
    printLog(nodeId, "<EM>: Forward:  Time per stage:");
    for (size_t i = 0; i < numLayers; ++i) {
        printLog(nodeId, "<EM>    Aggregation   %2d  %.3lf ms", i, vecTimeAggregate[i]);
        printLog(nodeId, "<EM>    Lambda        %2d  %.3lf ms", i, vecTimeLambda[i]);
        printLog(nodeId, "<EM>    Ghost update  %2d  %.3lf ms", i, vecTimeSendout[i]);
    }
    printLog(nodeId, "<EM>: Total forward-prop time %.3lf ms", timeForwardProcess);

    printLog(nodeId, "<EM>: Backward: Time per stage:");
    for (size_t i = numLayers; i < 2 * numLayers; i++) {
        printLog(nodeId, "<EM>    Aggregation   %2d  %.3lf ms", i, vecTimeAggregate[i]);
        printLog(nodeId, "<EM>    Lambda        %2d  %.3lf ms", i, vecTimeLambda[i]);
        printLog(nodeId, "<EM>    Ghost update  %2d  %.3lf ms", i, vecTimeSendout[i]);
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
Engine::readLayerConfigFile(std::string& layerConfigFileName) {
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
Engine::readFeaturesFile(std::string& featuresFileName) {
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
    while (infile.read(reinterpret_cast<char *> (&feature_vec[0]) , sizeof(FeatType) * featDim)) {
        // Set the vertex's initial values, if it is one of my local vertices / ghost vertices.
        if (graph.containsInEdgeGhostVertex(gvid)) {      // Global vertex.
            FeatType *actDataPtr = getVtxFeat(forwardGhostInitData, graph.getInEdgeGhostVertex(gvid).getLocalId(), featDim);
            memcpy(actDataPtr, feature_vec.data(), featDim * sizeof(FeatType));
        } else if (graph.containsVertex(gvid)) {    // Local vertex.
            FeatType *zDataPtr = getVtxFeat(localVerticesZData[0], graph.getVertexByGlobal(gvid).getLocalId(), featDim);
            FeatType *actDataPtr = getVtxFeat(localVerticesActivationData[0], graph.getVertexByGlobal(gvid).getLocalId(), featDim);
            memcpy(zDataPtr, feature_vec.data(), featDim * sizeof(FeatType));
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
Engine::readLabelsFile(std::string& labelsFileName) {
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

    while (infile.read(reinterpret_cast<char *> (&curr) , sizeof(unsigned))) {

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
Engine::readPartsFile(std::string& partsFileName, Graph& lGraph) {
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
Engine::processEdge(unsigned& from, unsigned& to, Graph& lGraph, std::set<unsigned> *inTopics, std::set<unsigned> *oTopics, int **forwardGhostVTables, int **backwardGhostVTables) {
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

            if (forwardGhostVTables[toPartition][lFromId] == 0) {
                forwardGhostVTables[toPartition][lFromId] = 1;
                forwardGhostVCnts[toPartition]++;
            }

            if (oTopics != NULL)
                oTopics->insert(from);
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

            if (backwardGhostVTables[fromPartition][lToId] == 0) {
                backwardGhostVTables[fromPartition][lToId] = 1;
                backwardGhostVCnts[fromPartition]++;
            }

            if (inTopics != NULL)
                inTopics->insert(from);
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
    for (Vertex& vertex : graph.getVertices()) {
        unsigned vtxDeg = vertex.getNumInEdges() + 1;
        float vtxNorm = std::pow(vtxDeg, -.5);
        vertex.setNormFactor(vtxNorm * vtxNorm);
        for (unsigned i = 0; i < vertex.getNumInEdges(); ++i) {
            InEdge& e = vertex.getInEdge(i);
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
            OutEdge& e = vertex.getOutEdge(i);
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
Engine::findGhostDegrees(std::string& fileName) {
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
Engine::readGraphBS(std::string& fileName, std::set<unsigned>& inTopics, std::vector<unsigned>& outTopics) {

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

    const unsigned graphLocalVerticesNum = graph.getNumLocalVertices();
    int **forwardGhostVTables = new int*[numNodes];
    int **backwardGhostVTables = new int*[numNodes];
    for (unsigned i = 0; i < numNodes; ++i) {
        if (i == nodeId) {
            continue;
        }
        forwardGhostVTables[i] = new int[graphLocalVerticesNum];
        backwardGhostVTables[i] = new int[graphLocalVerticesNum];
        memset(forwardGhostVTables[i], 0, sizeof(int) * graphLocalVerticesNum);
        memset(backwardGhostVTables[i], 0, sizeof(int) * graphLocalVerticesNum);
    }

    // Loop through all edges and process them.
    std::set<unsigned> oTopics;
    unsigned srcdst[2];
    while (infile.read((char *) srcdst, bSHeader.sizeOfVertexType * 2)) {
        if (srcdst[0] == srcdst[1])
            continue;

        processEdge(srcdst[0], srcdst[1], graph, &inTopics, &oTopics, forwardGhostVTables, backwardGhostVTables);
        if (undirected)
            processEdge(srcdst[1], srcdst[0], graph, &inTopics, &oTopics, forwardGhostVTables, backwardGhostVTables);
        graph.incrementNumGlobalEdges();
    }

    for (unsigned i = 0; i < numNodes; ++i) {
        if (i == nodeId) {
            forwardBatchMsgBuf[i] = NULL;
            backwardBatchMsgBuf[i] = NULL;
            continue;
        }
        forwardBatchMsgBuf[i] = new unsigned[forwardGhostVCnts[i]];
        backwardBatchMsgBuf[i] = new unsigned[backwardGhostVCnts[i]];
        unsigned forwardCnt = 0, backwardCnt = 0;
        for (unsigned j = 0; j < graphLocalVerticesNum; ++j) {
            if (forwardGhostVTables[i][j]) {
                forwardBatchMsgBuf[i][forwardCnt] = j;
                forwardCnt++;
            }
            if (backwardGhostVTables[i][j]) {
                backwardBatchMsgBuf[i][backwardCnt] = j;
                backwardCnt++;
            }
        }
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

    typename std::set<unsigned>::iterator it;
    for (it = oTopics.begin(); it != oTopics.end(); ++it)
        outTopics.push_back(*it);
}

// Substantiate an Engine object here for TF aggregators
Engine engine;