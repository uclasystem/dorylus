#include "../engine.hpp"
#include "../../utils/utils.hpp"

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


void Engine::aggregateGAT(Chunk &c) {
    unsigned start = c.lowBound;
    unsigned end = c.upBound;
    PROP_TYPE dir = c.dir;

    unsigned featDim;
    FeatType **inputFTensor = NULL; // forward edge activations
    FeatType **inputBTensor = NULL; // backward edge gradients
    FeatType *outputTensor = NULL;
    // Forward specific
    FeatType *featTensor = NULL;
    // Backward specific
    FeatType *dA = NULL;

    if (dir == PROP_TYPE::FORWARD) { // forward
        featDim = getFeatDim(c.layer);
        featTensor = getVtxFeat(savedNNTensors[c.layer - 1]["z"].getData(),
                                start, featDim);
        inputFTensor = savedEdgeTensors[c.layer - 1]["fedge"];
        outputTensor = savedNNTensors[c.layer - 1]["ah"].getData();
    } else { // backward
        featDim = getFeatDim(c.layer);
        inputFTensor = savedEdgeTensors[c.layer - 1]["fedge"];
        inputBTensor = savedEdgeTensors[c.layer - 1]["bedge"];
        outputTensor = savedNNTensors[c.layer - 1]["aTg"].getData();
        dA = savedNNTensors[c.layer - 1]["dA"].getData();
    }

    if (dir == PROP_TYPE::FORWARD) {
        FeatType *chunkPtr = getVtxFeat(outputTensor, start, featDim);
        std::memcpy(chunkPtr, featTensor,
                    sizeof(FeatType) * (end - start) * featDim);
    }
#ifdef _CPU_ENABLED_
#pragma omp parallel for
#endif
    for (unsigned lvid = start; lvid < end; lvid++) {
        // Read out data of the current layer of given vertex.
        FeatType *currDataDst = getVtxFeat(outputTensor, lvid, featDim);
        if (dir == PROP_TYPE::FORWARD) {
            // Aggregate activations from incoming neighbors.
            for (uint64_t eid = graph.forwardAdj.columnPtrs[lvid];
                eid < graph.forwardAdj.columnPtrs[lvid + 1]; ++eid) {
                EdgeType edgeWeight = graph.forwardAdj.values[eid];
                for (unsigned j = 0; j < featDim; ++j) {
                    currDataDst[j] += inputFTensor[eid][j] * edgeWeight;
                }
            }
        } else {
            // Aggregate gradients from outgoing neighbors.
            for (uint64_t eid = graph.backwardAdj.rowPtrs[lvid];
                eid < graph.backwardAdj.rowPtrs[lvid + 1]; ++eid) {
                EdgeType edgeWeight = graph.backwardAdj.values[eid];
                for (unsigned j = 0; j < featDim; ++j) {
                    currDataDst[j] += inputBTensor[eid][j] * edgeWeight;
                }
            }
            // Aggregate activations from incoming neighbors.
            for (uint64_t eid = graph.forwardAdj.columnPtrs[lvid];
                eid < graph.forwardAdj.columnPtrs[lvid + 1]; ++eid) {
                // Note the edgeGrad here is different from other for loops
                EdgeType edgeGrad = dA[eid];
                for (unsigned j = 0; j < featDim; ++j) {
                    currDataDst[j] += inputFTensor[eid][j] * edgeGrad;
                }
            }
        }
    }
}

// Get prediction after last forward layer of GAT
void Engine::predictGAT(Chunk &c) {
    unsigned featLayer = c.layer - 1;
    Matrix& labels = savedNNTensors[featLayer]["lab"];
    FeatType* labelPtr = labels.get(c.lowBound);
    FeatType* outputDeriv = savedNNTensors[featLayer]["grad"].get(c.lowBound);
    FeatType* agg = savedNNTensors[featLayer]["az"].get(c.lowBound);


    unsigned rows = c.upBound - c.lowBound;
    unsigned cols = labels.getCols();
    softmax(agg, outputDeriv, rows, cols);

#if defined(_CPU_ENABLED_) || defined(_GPU_ENABLED_)
#pragma omp parallel
#endif
    for (unsigned i = 0; i < rows * cols; ++i) {
        outputDeriv[i] -= labelPtr[i];
    }
}

void Engine::applyVertexGAT(Chunk &c) {
    c.vertex = true;
    if (c.dir == PROP_TYPE::FORWARD) {
        resComm->NNCompute(c);
    } else { // Backward, inc layer first before AVB
        Chunk nextChunk = incLayerGAT(c);
        resComm->NNCompute(nextChunk);
    }
}

void Engine::scatterGAT(Chunk &c) {
    unsigned outputLayer = c.layer - 1;
    unsigned featLayer = c.layer;
    std::string tensorName;
    if (c.dir == PROP_TYPE::FORWARD) {
        tensorName = "z";
    } else {
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
#if false || defined(_CPU_ENABLED_) || defined(_GPU_ENABLED_)
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

void Engine::applyEdgeGAT(Chunk &c) {
    c.vertex = false;
    resComm->NNCompute(c);
}