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

#include "../engine.hpp"
#include "../../utils/utils.hpp"

#ifdef _GPU_ENABLED_
#include "../../GPU-Computation/comp_unit.cuh"
#endif

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
    for (int layer = 0; layer < numLayers; ++layer) {
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
    for (int layer = numLayers - 1; layer > 0; --layer) {
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

#ifdef _GPU_ENABLED_
void Engine::aggregateGCN(Chunk &c) {
    PROP_TYPE dir = c.dir;

    // No edge feat tensor support for GPU. use featTensor + ghostTensor instead
    Matrix featTensor;
    Matrix ghostTensor;
    Matrix outputTensor;
    if (dir == PROP_TYPE::FORWARD) { // forward
        featTensor = c.layer == 0
                   ? savedNNTensors[c.layer]["x"]
                   : savedNNTensors[c.layer - 1]["h"];
        ghostTensor = savedNNTensors[c.layer]["fg"];
        outputTensor = savedNNTensors[c.layer]["ah"]; // output aggregatedTensor
    } else { // backward
        featTensor = savedNNTensors[c.layer]["grad"];
        ghostTensor = savedNNTensors[c.layer - 1]["bg"];
        outputTensor = savedNNTensors[c.layer - 1]["aTg"];
    }
    CuMatrix feat;
    feat.loadSpDense(featTensor.getData(), ghostTensor.getData(),
                     featTensor.getRows(), ghostTensor.getRows(),
                     featTensor.getCols());
    CuMatrix out;
    if (dir == PROP_TYPE::FORWARD) {
        out = cu.aggregate(*NormAdjMatrixIn, feat, *OneNorms);
    } else {
        out = cu.aggregate(*NormAdjMatrixOut, feat, *OneNorms);
    }
    out.setData(outputTensor.getData());
    out.updateMatrixFromGPU();
    cudaDeviceSynchronize();
    CuMatrix::freeGPU();
}
#else // !defined(_GPU_ENABLED_)
void Engine::aggregateGCN(Chunk &c) {
    unsigned start = c.lowBound;
    unsigned end = c.upBound;
    PROP_TYPE dir = c.dir;

    unsigned featDim;
    FeatType *featTensor = NULL;
    FeatType **inputTensor = NULL;
    FeatType *outputTensor = NULL;
    if (dir == PROP_TYPE::FORWARD) { // forward
        featDim = getFeatDim(c.layer);
        featTensor = c.layer == 0
                   ? getVtxFeat(savedNNTensors[c.layer]["x"].getData(),
                                start, featDim)
                   : getVtxFeat(savedNNTensors[c.layer - 1]["h"].getData(),
                                start, featDim);
        inputTensor = savedEdgeTensors[c.layer]["fedge"];  // input edgeFeatsTensor
        outputTensor = savedNNTensors[c.layer]["ah"].getData(); // output aggregatedTensor
    } else { // backward
        featDim = getFeatDim(c.layer);
        featTensor = getVtxFeat(savedNNTensors[c.layer]["grad"].getData(),
                                start, featDim);
        inputTensor = savedEdgeTensors[c.layer - 1]["bedge"];
        outputTensor = savedNNTensors[c.layer - 1]["aTg"].getData();
    }
    FeatType *chunkPtr = getVtxFeat(outputTensor, start, featDim);
    std::memcpy(chunkPtr, featTensor,
                sizeof(FeatType) * (end - start) * featDim);

#ifdef _CPU_ENABLED_
#pragma omp parallel for
#endif
    for (unsigned lvid = start; lvid < end; lvid++) {
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
        if (dir == PROP_TYPE::FORWARD) { // using forward adj mat
            for (uint64_t eid = graph.forwardAdj.columnPtrs[lvid];
                eid < graph.forwardAdj.columnPtrs[lvid + 1]; ++eid) {
                EdgeType normFactor = graph.forwardAdj.values[eid];
                for (unsigned j = 0; j < featDim; ++j) {
                    currDataDst[j] += inputTensor[eid][j] * normFactor;
                }
            }
        } else { // using backward adj mat
            for (uint64_t eid = graph.backwardAdj.rowPtrs[lvid];
                eid < graph.backwardAdj.rowPtrs[lvid + 1]; ++eid) {
                EdgeType normFactor = graph.backwardAdj.values[eid];
                for (unsigned j = 0; j < featDim; ++j) {
                    currDataDst[j] += inputTensor[eid][j] * normFactor;
                }
            }
        }
    }
}
#endif // _GPU_ENABLED

void Engine::applyVertexGCN(Chunk &c) {
    c.vertex = true;
    if (c.dir == PROP_TYPE::FORWARD) { // Forward pass
        resComm->NNCompute(c);
    } else { // Backward pass, inc layer first to align up the layer id
        Chunk nextC = incLayerGCN(c);
        resComm->NNCompute(nextC);
    }
}

void Engine::scatterGCN(Chunk &c) {
    unsigned outputLayer = c.layer;
    std::string tensorName;
    if (c.dir == PROP_TYPE::FORWARD) {
        outputLayer -= 1;
        tensorName = "h";
    } else {
        tensorName = "grad";
    }
    FeatType *scatterTensor =
        savedNNTensors[outputLayer][tensorName].getData();

    unsigned startId = c.lowBound;
    unsigned endId = c.upBound;
    unsigned featDim = getFeatDim(c.layer);

    std::map<unsigned, std::vector<unsigned>> &ghostMap =
        c.dir == PROP_TYPE::FORWARD ? graph.forwardGhostMap
                                    : graph.backwardGhostMap;

    // batch sendouts similar to the sequential version
    const unsigned BATCH_SIZE = std::max(
        (MAX_MSG_SIZE - DATA_HEADER_SIZE) /
            (sizeof(unsigned) + sizeof(FeatType) * featDim),
        1ul);  // at least send one vertex
    // Create a series of buckets for batching sendout messages to nodes
    auto *batchedIds = new std::vector<unsigned>[numNodes];
    for (unsigned lvid = startId; lvid < endId; ++lvid) {
        for (unsigned nid : ghostMap[lvid]) {
            batchedIds[nid].push_back(lvid);
        }
    }

    for (unsigned nid = 0; nid < numNodes; ++nid) {
        if (nid == nodeId)
            continue;
        unsigned ghostVCnt = batchedIds[nid].size();
#if defined(_GPU_ENABLED_)
#pragma omp parallel for
#endif
        for (unsigned ib = 0; ib < ghostVCnt; ib += BATCH_SIZE) {
            unsigned sendBatchSize = (ghostVCnt - ib) < BATCH_SIZE
                                   ? (ghostVCnt - ib) : BATCH_SIZE;
            verticesPushOut(nid, sendBatchSize,
                            batchedIds[nid].data() + ib, scatterTensor,
                            featDim, c);
            if (!async) {
                // recvCntLock.lock();
                // recvCnt++;
                // recvCntLock.unlock();
                __sync_fetch_and_add(&recvCnt, 1);
            }
        }
    }

    delete[] batchedIds;
}

void Engine::ghostReceiverGCN(unsigned tid) {
    // printLog(nodeId, "RECEIVER: Starting");
    BackoffSleeper bs;
    unsigned sender, topic;
    FeatType *msgBuf = (FeatType *)new char[MAX_MSG_SIZE];

    // While loop, looping infinitely to get the next message.
    while (true) {
        // No message in queue.
        if (!commManager.dataPullIn(&sender, &topic, msgBuf, MAX_MSG_SIZE)) {
            bs.sleep();
            if (pipelineHalt) {
                break;
            }
            // Pull in the next message, and process this message.
        } else {
            // A normal ghost value broadcast.
            if (topic < MAX_IDTYPE - 1) {
                if (!async) {
                    // Using MAX_IDTYPE - 1 as the receive signal.
                    commManager.dataPushOut(sender, nodeId, MAX_IDTYPE - 1, NULL, 0);
                }
                char *bufPtr = (char *)msgBuf;
                unsigned recvGhostVCnt = topic;
                unsigned featDim = *(unsigned *)bufPtr;
                bufPtr += sizeof(unsigned);
                unsigned layer = *(unsigned *)bufPtr;
                bufPtr += sizeof(unsigned);
                unsigned dir = *(unsigned *)bufPtr;
                bufPtr += sizeof(unsigned);
                // Get proper variables depending on forward or backward
                std::string tensorName = dir == PROP_TYPE::FORWARD
                                       ? "fg" : "bg";
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

                if (!async) {
                    // recvCntLock.lock();
                    // ghostVtcsRecvd += topic;
                    // recvCntLock.unlock();
                    __sync_fetch_and_add(&ghostVtcsRecvd, topic);
                }

                // A respond to a broadcast, and the topic vertex is in my local
                // vertices. I should update the corresponding recvWaiter's
                // value. If waiters become empty, send a signal in case the
                // workers are waiting on it to be empty at the layer barrier.
            } else { // (topic == MAX_IDTYPE - 1)
                if (!async) {
                    // recvCntLock.lock();
                    // recvCnt--;
                    // recvCntLock.unlock();
                    __sync_fetch_and_add(&recvCnt, -1);
                }
            }
            unsigned totalGhostCnt = currDir == PROP_TYPE::FORWARD
                                   ? graph.srcGhostCnt
                                   : graph.dstGhostCnt;

            if (!async) {
                recvCntLock.lock();
                if (recvCnt == 0 && ghostVtcsRecvd == totalGhostCnt) {
                    recvCntCond.signal();
                }
                recvCntLock.unlock();
            }

            bs.reset();
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

void Engine::applyEdgeGCN(Chunk &chunk) {
    GAQueue.push_atomic(chunk);
}
