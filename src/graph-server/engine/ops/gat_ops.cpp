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

void Engine::preallocateGAT() {
    unsigned vtxCnt = graph.localVtxCnt;

    // Store input tesnors
    savedNNTensors[0]["h"] =
        Matrix(vtxCnt, getFeatDim(0), forwardVerticesInitData);

    // savedNNTensors[0]["fg"] =
    //     Matrix(graph.srcGhostCnt, getFeatDim(0), forwardGhostInitData);
    savedNNTensors[numLayers - 1]["lab"] =
        Matrix(vtxCnt, getFeatDim(numLayers), localVerticesLabels);

    // forward tensor allocation
    for (int layer = 0; layer < numLayers; ++layer) {
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
        // FeatType **edgeTensor =
        //     srcVFeats2eFeats(hTensor, ghostTensor, vtxCnt, featDim);
        // savedEdgeTensors[layer + 1]["fedge"] = edgeTensor;
    }

    // backward tensor allocation
    for (int layer = numLayers - 1; layer >= 0; --layer) {
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

#ifdef _GPU_ENABLED_
void Engine::aggregateGAT(Chunk &c) {
    if (c.dir == PROP_TYPE::FORWARD) { // forward
        Matrix &featTensor = savedNNTensors[c.layer - 1]["z"];
        Matrix &ghostTensor = savedNNTensors[c.layer - 1]["fg_z"];
        Matrix &outputTensor = savedNNTensors[c.layer - 1]["ah"];
        Matrix &A = savedNNTensors[c.layer - 1]["A"];

        CuMatrix feat;
        feat.loadSpDense(featTensor.getData(), ghostTensor.getData(),
                         featTensor.getRows(), ghostTensor.getRows(),
                         featTensor.getCols());
        CuMatrix dummy_e = cu.wrapMatrix(A); // getting A into VRAM
        CuMatrix e = *NormAdjMatrixIn;
        // YIFAN: VRAM won't leak since dummy_e will be freed by CuMatrix::freeGPU
        e.csrVal = dummy_e.devPtr;
        CuMatrix out = cu.aggregate(e, feat, *OneNorms);
        // printLog(nodeId, "Out shape: %s", out.shape().c_str());
        out.setData(outputTensor.getData());
        out.updateMatrixFromGPU();
        cudaDeviceSynchronize();
        CuMatrix::freeGPU();
    } else { // backward
        Matrix &fFeatTensor = savedNNTensors[c.layer - 1]["z"];
        Matrix &fGhostTensor = savedNNTensors[c.layer - 1]["fg_z"];
        Matrix &bFeatTensor = savedNNTensors[c.layer - 1]["grad"];
        Matrix &bGhostTensor = savedNNTensors[c.layer - 1]["bg_d"];
        Matrix &outputTensor = savedNNTensors[c.layer - 1]["aTg"];
        Matrix &dmdA = savedNNTensors[c.layer - 1]["dA"]; // for dummydA

        CuMatrix z;
        z.loadSpDense(fFeatTensor.getData(), fGhostTensor.getData(),
                      fFeatTensor.getRows(), fGhostTensor.getRows(),
                      fFeatTensor.getCols());
        // (1) dA.dot(Z)
        CuMatrix dummydA = cu.wrapMatrix(dmdA); // getting dA into VRAM
        CuMatrix dA = *NormAdjMatrixIn;
        dA.csrVal = dummydA.devPtr;
        CuMatrix dAZ = cu.aggregate(dA, z, *ZeroNorms);
        // (2) A.transpose().dot(dP)
        CuMatrix dP;
        dP.loadSpDense(bFeatTensor.getData(), bGhostTensor.getData(),
                       bFeatTensor.getRows(), bGhostTensor.getRows(),
                       bFeatTensor.getCols());
        CuMatrix A = *NormAdjMatrixOut;
        CuMatrix AdP = cu.aggregate(A, dP, *ZeroNorms);
        cu.hadamardAdd(AdP, dAZ);
        CuMatrix res = AdP;
        // printLog(nodeId, "Out shape: %s", res.shape().c_str());
        res.setData(outputTensor.getData());
        res.updateMatrixFromGPU();
        cudaDeviceSynchronize();
        CuMatrix::freeGPU();
    }
}
#else // !defined(_GPU_ENABLED_)
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
            // A.transpose().dot(dPred)
            // Aggregate gradients from outgoing neighbors.
            for (uint64_t eid = graph.backwardAdj.rowPtrs[lvid];
                eid < graph.backwardAdj.rowPtrs[lvid + 1]; ++eid) {
                EdgeType edgeWeight = graph.backwardAdj.values[eid];
                for (unsigned j = 0; j < featDim; ++j) {
                    currDataDst[j] += inputBTensor[eid][j] * edgeWeight;
                }
            }
            // dA.dot(Z)
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
#endif // _GPU_ENABLED_

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
#if false && (defined(_CPU_ENABLED_) || defined(_GPU_ENABLED_))
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

void Engine::ghostReceiverGAT(unsigned tid) {
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
                                       ? "fg_z" : "bg_d";
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

void Engine::applyEdgeGAT(Chunk &c) {
    c.vertex = false;
    resComm->NNCompute(c);
}