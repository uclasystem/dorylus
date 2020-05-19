#include "../engine.hpp"

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


FeatType **Engine::scatter(FeatType *vtcsTensor, unsigned vtcsCnt,
                           unsigned featDim) {
    double sttTimer = getTimer();

    // Start data communicators.
    commHalt = false;
    recvCnt = 0;
    forwardGhostVerticesDataOut = savedNNTensors[layer]["fg"].getData();
    if (forwardGhostVerticesDataOut == NULL) {
        printLog(nodeId, "Forward scatter buffer pointer is NULL");
    }
    auto fgr_fp =
        std::bind(&Engine::forwardGhostReceiver, this, std::placeholders::_1);
    dataPool->perform(fgr_fp);

    sendForwardGhostUpdates(vtcsTensor, featDim);

    // TODO: (YIFAN) we can optimize this to extend comm protocol. Mark the last
    // packet sent so this node knows when to exit ghostCommunicator.
    nodeManager.barrier();

    commHalt = true;
    // Join all data communicators.
    dataPool->sync();

    // FeatType **edgsTensor = srcVFeats2eFeats(vtcsTensor,
    // forwardGhostVerticesDataIn, vtcsCnt, featDim);
    FeatType **edgsTensor = savedEdgeTensors[layer]["fedge"];
    vtcsTensor = NULL;

    if (vecTimeScatter.size() < numLayers) {
        vecTimeScatter.push_back(getTimer() - sttTimer);
    } else {
        vecTimeScatter[layer] += getTimer() - sttTimer;
    }

    return edgsTensor;
}

FeatType **Engine::scatterBackward(FeatType *gradTensor, unsigned vtcsCnt,
                                   unsigned featDim) {
    double sttTimer = getTimer();

    // Start data communicators.
    commHalt = false;
    recvCnt = 0;
    // YIFAN: Do we really need reset bkwdRecvCnt? Same question for all 3 reset
    // JOHN: Yifan, no we don't. I implemented that when I suspected that
    //  having a single counter for fwd and bkwd was the problem with pipelining
    backwardGhostVerticesDataOut = savedNNTensors[layer - 1]["bg"].getData();
    auto bgr_fp =
        std::bind(&Engine::backwardGhostReceiver, this, std::placeholders::_1);
    dataPool->perform(bgr_fp);

    sendBackwardGhostGradients(gradTensor, featDim);

    //## Global Iteration barrier. ##/
    // TODO: (YIFAN) we can optimize this to extend comm protocal. Mark the last
    // packet sent so this node knows when to exit ghostCommunicator.
    nodeManager.barrier();
    commHalt = true;
    // Join all data communicators.
    dataPool->sync();

    // FeatType **eFeats = dstVFeats2eFeats(gradTensor,
    // backwardGhostVerticesDataIn, vtcsCnt, featDim);
    FeatType **eFeats = savedEdgeTensors[layer - 1]["bedge"];
    gradTensor = NULL;

    if (vecTimeScatter.size() < 2 * numLayers) {
        for (unsigned i = vecTimeScatter.size(); i < 2 * numLayers; i++) {
            vecTimeScatter.push_back(0.0);
        }
    }
    vecTimeScatter[numLayers + layer - 1] += getTimer() - sttTimer;
    return eFeats;
}

/**
 *
 * Major part of the engine's communication logic is done by data threads.
 * These threads loop asynchronously with computation workers.
 *
 */
void Engine::forwardGhostReceiver(unsigned tid) {
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
            // Computation workers done their work, so communicator goes to
            // death as well.
            if (commHalt) {
                delete[] msgBuf;
                if (commManager.dataPullIn(&sender, &topic, msgBuf,
                                           MAX_MSG_SIZE)) {
                    printLog(
                        nodeId,
                        "\033[1;31m[ ERROR ]\033[0m Still messages in buffer");
                }

                return;
            }

            usleep(SLEEP_PERIOD);  // sleep a little and give up CPUs
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
                commManager.dataPushOut(sender, nodeId, MAX_IDTYPE - 1, NULL,
                                        0);
                vtcsRecvd += topic;

                char *bufPtr = (char *)msgBuf;
                unsigned recvGhostVCnt = topic;
                // Update ghost vertices
                for (unsigned i = 0; i < recvGhostVCnt; ++i) {
                    unsigned gvid = *(unsigned *)bufPtr;
                    bufPtr += sizeof(unsigned);
                    FeatType *dataPtr = getVtxFeat(
                        forwardGhostVerticesDataOut,
                        graph.srcGhostVtcs[gvid] - graph.localVtxCnt, featDim);
                    memcpy(dataPtr, bufPtr, sizeof(FeatType) * featDim);
                    bufPtr += sizeof(FeatType) * featDim;
                }

                // A respond to a broadcast, and the topic vertex is in my local
                // vertices. I should update the corresponding recvWaiter's
                // value. If waiters become empty, send a signal in case the
                // workers are waiting on it to be empty at the layer barrier.
            } else {  // (topic == MAX_IDTYPE - 1)
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


/**
 *
 * Major part of the engine's communication logic is done by data threads.
 * These threads loop asynchronously with computation workers.
 *
 */
void Engine::backwardGhostReceiver(unsigned tid) {
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
            // Computation workers done their work, so communicator goes to
            // death as well.
            if (commHalt) {
                delete[] msgBuf;
                // Better to use return than break for compiler optimization
                return;
            }

            usleep(SLEEP_PERIOD);  // sleep a little and give up CPUs
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
                commManager.dataPushOut(sender, nodeId, MAX_IDTYPE - 1, NULL,
                                        0);
                vtcsRecvd += topic;

                char *bufPtr = (char *)msgBuf;
                unsigned recvGhostVCnt = topic;
                // Update ghost vertices
                for (unsigned i = 0; i < recvGhostVCnt; ++i) {
                    unsigned gvid = *(unsigned *)bufPtr;
                    bufPtr += sizeof(unsigned);
                    FeatType *dataPtr = getVtxFeat(
                        backwardGhostVerticesDataOut,
                        graph.dstGhostVtcs[gvid] - graph.localVtxCnt, featDim);
                    memcpy(dataPtr, bufPtr, sizeof(FeatType) * featDim);
                    bufPtr += sizeof(FeatType) * featDim;
                }

                // A respond to a broadcast, and the topic vertex is in my local
                // vertices. I should update the corresponding recvWaiter's
                // value. If waiters become empty, send a signal in case the
                // workers are waiting on it to be empty at the layer barrier.
            } else {  // (topic == MAX_IDTYPE - 1)
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


// async pipeline
void Engine::scatterWorker(unsigned tid) {
    // printLog(nodeId, "SCATTER: Starting");
    BackoffSleeper bs;

    // Check queue to see if partition ready
    while (true) {
        scatQueueLock.lock();
        if (scatterQueue.empty()) {
            scatQueueLock.unlock();

            if (commHalt) {
                break;
            }
            // sleep with backoff
            bs.sleep();
        } else {
            Chunk c = scatterQueue.top();
            scatterQueue.pop();
            scatQueueLock.unlock();

            unsigned startScat = timestamp_ms();

            // printLog(nodeId, "SCATTER: Got %s", c.str().c_str());

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

            FeatType *scatterTensor =
                savedNNTensors[outputLayer][tensorName].getData();

            unsigned startId = c.lowBound;
            unsigned endId = c.upBound;
            unsigned featDim = getFeatDim(featLayer);

            PROP_TYPE dir = c.dir;
            std::map<unsigned, std::vector<unsigned>> &ghostMap =
                dir == PROP_TYPE::FORWARD ? graph.forwardGhostMap
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
            bool batchFlag = true;
            unsigned BATCH_SIZE = std::max(
                ((batchFlag ? MAX_MSG_SIZE : 4096) - DATA_HEADER_SIZE) /
                    (sizeof(unsigned) + sizeof(FeatType) * featDim),
                1ul);  // at least send one vertex
            for (unsigned nid = 0; nid < numNodes; ++nid) {
                if (nid == nodeId) {
                    continue;
                }

                unsigned ghostVCnt = batchedIds[nid].size();
                for (unsigned ib = 0; ib < ghostVCnt; ib += BATCH_SIZE) {
                    unsigned sendBatchSize = (ghostVCnt - ib) < BATCH_SIZE
                                                 ? (ghostVCnt - ib)
                                                 : BATCH_SIZE;

                    verticesPushOut(nid, sendBatchSize,
                                    batchedIds[nid].data() + ib, scatterTensor,
                                    featDim, c);
                }
            }
            unsigned endScat = timestamp_ms();
            // Add chunk into appropriate aggregate queue
            // printLog(nodeId, "SCATTER: Finished %s", c.str().c_str());
            aggQueueLock.lock();
            aggregateQueue.push(c);
            vecTimeScatter[c.dir * numLayers + c.layer] += endScat - startScat;
            aggQueueLock.unlock();

            delete[] batchedIds;
            bs.reset();
        }
    }

    // clean up remaining chunks
    scatQueueLock.lock();
    if (!scatterQueue.empty()) {
        printLog(nodeId, "CLEAN UP: Scatter queue is not empty!");
    }
    while (!scatterQueue.empty()) {
        scatterQueue.pop();
    }
    scatQueueLock.unlock();
}

void Engine::ghostReceiver(unsigned tid) {
    // printLog(nodeId, "RECEIVER: Starting");
    // backoff sleep strategy to improve CPU utilization
    int failedTrials = 0;
    const int INIT_PERIOD = 256;
    const int MAX_PERIOD = 4096;
    int SLEEP_PERIOD = INIT_PERIOD;
    unsigned sender, topic;
    std::string tensorName;
    FeatType *msgBuf = (FeatType *)new char[MAX_MSG_SIZE];

    // While loop, looping infinitely to get the next message.
    while (true) {
        // No message in queue.
        if (!commManager.dataPullIn(&sender, &topic, msgBuf, MAX_MSG_SIZE)) {
            usleep(SLEEP_PERIOD);  // sleep a little and give up CPUs
            failedTrials++;
            if (failedTrials == 64 && SLEEP_PERIOD < MAX_PERIOD) {
                failedTrials = 0;
                SLEEP_PERIOD *= 2;
            }
            if (commHalt) {
                break;
            }
            // Pull in the next message, and process this message.
        } else {
            // A normal ghost value broadcast.
            if (topic < MAX_IDTYPE - 1) {
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
                std::map<unsigned, unsigned> &globalToGhostVtcs =
                    dir == PROP_TYPE::FORWARD ? graph.srcGhostVtcs
                                              : graph.dstGhostVtcs;

                // printLog(nodeId, "RECEIVER: Got msg %u:%s", layer,
                //   dir == PROP_TYPE::FORWARD ? "F" : "B");
                FeatType *ghostData =
                    savedNNTensors[layer][tensorName].getData();
                if (ghostData == NULL) {
                    printLog(nodeId,
                             "RECEIVER: Coudn't find tensor '%s' for layer %u",
                             tensorName.c_str(), layer);
                }

                // Update ghost vertices
                for (unsigned i = 0; i < recvGhostVCnt; ++i) {
                    unsigned gvid = *(unsigned *)bufPtr;
                    bufPtr += sizeof(unsigned);
                    FeatType *dataPtr = getVtxFeat(
                        ghostData, globalToGhostVtcs[gvid] - graph.localVtxCnt,
                        featDim);
                    memcpy(dataPtr, bufPtr, sizeof(FeatType) * featDim);
                    bufPtr += sizeof(FeatType) * featDim;
                }

                // A respond to a broadcast, and the topic vertex is in my local
                // vertices. I should update the corresponding recvWaiter's
                // value. If waiters become empty, send a signal in case the
                // workers are waiting on it to be empty at the layer barrier.
            } else {}  // (topic == MAX_IDTYPE - 1)

            SLEEP_PERIOD = INIT_PERIOD;
            failedTrials = 0;
        }
    }

    if (commManager.dataPullIn(&sender, &topic, msgBuf, MAX_MSG_SIZE)) {
        printLog(nodeId, "CLEAN UP: Still messages in buffer");
        // clean up
        while (commManager.dataPullIn(&sender, &topic, msgBuf,
                                        MAX_MSG_SIZE)) {};
    }
    delete[] msgBuf;
}

/////////////////////////////////////////////////////////
// Below are private forward functions for the engine. //
/////////////////////////////////////////////////////////

// Loop through all local vertices and do the data send out work.
// If there are any remote edges for a vertex, should send this vid to
// other nodes for their ghost's update.
inline void Engine::sendForwardGhostUpdates(FeatType *inputTensor,
                                            unsigned featDim) {
    bool batchFlag = true;
    unsigned BATCH_SIZE =
        std::max(((batchFlag ? MAX_MSG_SIZE : 4096) - DATA_HEADER_SIZE) /
                     (sizeof(unsigned) + sizeof(FeatType) * featDim),
                 1ul);  // at least send one vertex
    for (unsigned nid = 0; nid < numNodes; ++nid) {
        if (nid == nodeId) {
            continue;
        }

        unsigned forwardGhostVCnt = graph.forwardLocalVtxDsts[nid].size();
        for (unsigned ib = 0; ib < forwardGhostVCnt; ib += BATCH_SIZE) {
            unsigned sendBatchSize = (forwardGhostVCnt - ib) < BATCH_SIZE
                                         ? (forwardGhostVCnt - ib)
                                         : BATCH_SIZE;

            forwardVerticesPushOut(nid, sendBatchSize,
                                   graph.forwardLocalVtxDsts[nid].data() + ib,
                                   inputTensor, featDim);
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

inline void Engine::forwardVerticesPushOut(unsigned receiver, unsigned totCnt,
                                           unsigned *lvids,
                                           FeatType *inputTensor,
                                           unsigned featDim) {
    zmq::message_t msg(DATA_HEADER_SIZE +
                       (sizeof(unsigned) + sizeof(FeatType) * featDim) *
                           totCnt);
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
        FeatType *dataPtr = getVtxFeat(inputTensor, lvids[i], featDim);
        memcpy(msgPtr, dataPtr, sizeof(FeatType) * featDim);
        msgPtr += sizeof(FeatType) * featDim;
    }
    commManager.rawMsgPushOut(msg);
}

void Engine::verticesPushOut(unsigned receiver, unsigned totCnt,
                             unsigned *lvids, FeatType *inputTensor,
                             unsigned featDim, Chunk &c) {
    zmq::message_t msg(DATA_HEADER_SIZE +
                       (sizeof(unsigned) + sizeof(FeatType) * featDim) *
                           totCnt);
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


//////////////////////////////////////////////////////////
// Below are private backward functions for the engine. //
//////////////////////////////////////////////////////////
void Engine::sendBackwardGhostGradients(FeatType *gradTensor,
                                        unsigned featDim) {
    // Loop through all local vertices and do the data send out work. If there
    // are any remote edges for a vertex, should send this vid to other nodes
    // for their ghost's update.
    bool batchFlag = true;
    unsigned BATCH_SIZE =
        std::max(((batchFlag ? MAX_MSG_SIZE : 4096) - DATA_HEADER_SIZE) /
                     (sizeof(unsigned) + sizeof(FeatType) * featDim),
                 1ul);  // at least send one vertex
    for (unsigned nid = 0; nid < numNodes; ++nid) {
        if (nid == nodeId) {
            continue;
        }
        unsigned backwardGhostVCnt = graph.backwardLocalVtxDsts[nid].size();
        for (unsigned ib = 0; ib < backwardGhostVCnt; ib += BATCH_SIZE) {
            unsigned sendBatchSize = (backwardGhostVCnt - ib) < BATCH_SIZE
                                         ? (backwardGhostVCnt - ib)
                                         : BATCH_SIZE;

            backwardVerticesPushOut(nid, sendBatchSize,
                                    graph.backwardLocalVtxDsts[nid].data() + ib,
                                    gradTensor, featDim);
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

inline void Engine::backwardVerticesPushOut(unsigned receiver, unsigned totCnt,
                                            unsigned *lvids,
                                            FeatType *gradTensor,
                                            unsigned featDim) {
    zmq::message_t msg(DATA_HEADER_SIZE +
                       (sizeof(unsigned) + sizeof(FeatType) * featDim) *
                           totCnt);
    char *msgPtr = (char *)(msg.data());
    sprintf(msgPtr, NODE_ID_HEADER, receiver);
    msgPtr += NODE_ID_DIGITS;
    *(unsigned *)msgPtr = nodeId;
    msgPtr += sizeof(unsigned);
    *(unsigned *)msgPtr = totCnt;
    ;
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
// void Engine::pipelineGhostReceiver(unsigned tid) {
//     // backoff sleep strategy to improve CPU utilization
//     int failedTrials = 0;
//     const int INIT_PERIOD = 256;
//     const int MAX_PERIOD = 4096;
//     int SLEEP_PERIOD = INIT_PERIOD;
//     unsigned sender, topic;
//     unsigned vtcsRecvd = 0;
//     FeatType *msgBuf = (FeatType *)new char[MAX_MSG_SIZE];

//     // While loop, looping infinitely to get the next message.
//     while (!commHalt) {
//         // No message in queue.
//         if (!commManager.dataPullIn(&sender, &topic, msgBuf, MAX_MSG_SIZE)) {
//             // Computation workers done their work, so communicator goes to
//             // death as well.
//             if (commHalt) {
//                 delete[] msgBuf;
//                 if (commManager.dataPullIn(&sender, &topic, msgBuf,
//                                            MAX_MSG_SIZE)) {
//                     printLog(
//                         nodeId,
//                         "\033[1;31m[ ERROR ]\033[0m Still messages in buffer");
//                 }
//                 return;
//             }

//             usleep(SLEEP_PERIOD);  // sleep a little and give up CPUs
//             failedTrials++;
//             if (failedTrials == 64 && SLEEP_PERIOD < MAX_PERIOD) {
//                 failedTrials = 0;
//                 SLEEP_PERIOD *= 2;
//             }
//             // Pull in the next message, and process this message.
//         } else {
//             // A normal ghost value broadcast.
//             if (topic < MAX_IDTYPE - 1) {
//                 // Using MAX_IDTYPE - 1 as the receive signal.
//                 commManager.dataPushOut(sender, nodeId, MAX_IDTYPE - 1, NULL,
//                                         0);
//                 vtcsRecvd += topic;

//                 char *bufPtr = (char *)msgBuf;
//                 unsigned recvGhostVCnt = topic;
//                 unsigned featDim = getFeatDim(layer + 1);
//                 // Update ghost vertices
//                 for (unsigned i = 0; i < recvGhostVCnt; ++i) {
//                     unsigned gvid = *(unsigned *)bufPtr;
//                     bufPtr += sizeof(unsigned);
//                     FeatType *dataPtr = getVtxFeat(
//                         forwardGhostVerticesDataOut,
//                         graph.srcGhostVtcs[gvid] - graph.localVtxCnt, featDim);
//                     memcpy(dataPtr, bufPtr, sizeof(FeatType) * featDim);
//                     bufPtr += sizeof(FeatType) * featDim;
//                 }

//                 // A respond to a broadcast, and the topic vertex is in my local
//                 // vertices. I should update the corresponding recvWaiter's
//                 // value. If waiters become empty, send a signal in case the
//                 // workers are waiting on it to be empty at the layer barrier.
//             } else {  // (topic == MAX_IDTYPE - 1)
//                 recvCntLock.lock();
//                 recvCnt--;
//                 recvCntLock.unlock();
//             }
//             recvCntLock.lock();
//             if (recvCnt == 0 && vtcsRecvd == graph.srcGhostCnt) {
//                 recvCntCond.signal();
//             }
//             recvCntLock.unlock();

//             SLEEP_PERIOD = INIT_PERIOD;
//             failedTrials = 0;
//         }
//     }

//     if (commManager.dataPullIn(&sender, &topic, msgBuf, MAX_MSG_SIZE)) {
//         printLog(nodeId, "\033[1;31m[ ERROR ]\033[0m Still messages in buffer");
//     }

//     delete[] msgBuf;
// }
