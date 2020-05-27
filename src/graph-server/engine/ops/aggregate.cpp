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

#ifdef _GPU_ENABLED_
// #include "../../../common/utils.hpp"
#include "../../GPU-Computation/comp_unit.cuh"
CuMatrix *NormAdjMatrixIn = NULL;
CuMatrix *NormAdjMatrixOut = NULL;
CuMatrix *Norms = NULL;
ComputingUnit cu = ComputingUnit::getInstance();
#endif

FeatType *Engine::aggregate(FeatType **edgsTensor, unsigned edgsCnt,
                            unsigned featDim, AGGREGATOR aggregator) {
    double sttTimer = getTimer();
#ifdef _GPU_ENABLED_
    std::cout << "Start GPU aggregation forward\n";

    // Loading tensor in CPU
    FeatType *outputTensor = savedNNTensors[layer]["ah"].getData();
    FeatType *gTensor = savedNNTensors[layer]["fg_z"].getData();
    FeatType *hTensor = savedNNTensors[layer]["z"].getData();

    // Load Feature into GPU memory
    CuMatrix feat;
    feat.loadSpDense(hTensor, gTensor, graph.localVtxCnt, graph.srcGhostCnt,
                     featDim);

    switch (aggregator) {
        case (AGGREGATOR::WSUM): {
            CuMatrix dummy_e = cu.wrapMatrix(
                savedNNTensors[layer]["A"]);  // for getting A into VRAM
            CuMatrix e = *NormAdjMatrixIn;
            e.csrVal = dummy_e.devPtr;
            CuMatrix out = cu.aggregate(e, feat);
            std::cout << "Out shape: " << out.shape() << std::endl;
            out.setData(outputTensor);
            out.updateMatrixFromGPU();
            cudaDeviceSynchronize();
            CuMatrix::freeGPU();
            break;
        }
        default:
            printLog(nodeId, "Invalid Aggregator %d.", aggregator);
            break;
    }
    currId = graph.localVtxCnt;
#else
    // AH
    FeatType *hTensor = savedNNTensors[layer]["z"].getData();
    FeatType *outputTensor = savedNNTensors[layer]["ah"].getData();
    assert(hTensor != NULL);
    assert(outputTensor != NULL);

    currId = 0;

    switch (aggregator) {
        case (AGGREGATOR::WSUM): {
            memcpy(outputTensor, hTensor,
                   sizeof(FeatType) * graph.localVtxCnt * featDim);

            AggOPArgs args = {outputTensor, edgsTensor, graph.localVtxCnt,
                              edgsCnt, featDim};
            auto computeFn =
                std::bind(&Engine::aggregateCompute, this,
                          std::placeholders::_1, std::placeholders::_2);

            computePool->perform(computeFn, &args);
            computePool->sync();

            break;
        }
        default:
            printLog(nodeId, "Invalid Aggregator %d.", aggregator);
            break;
    }
#endif  // _GPU_ENABLED_

    vecTimeAggregate[layer] += getTimer() - sttTimer;
    return outputTensor;
}

/**
 *
 * Major part of the engine's backward-prop logic.
 *
 */
FeatType *Engine::aggregateBackward(FeatType **eVGradTensor, unsigned edgsCnt,
                                    unsigned featDim, AGGREGATOR aggregator) {
    double sttTimer = getTimer();
    currId = 0;
    FeatType *outputTensor = savedNNTensors[layer]["aTg"].getData();
    FeatType *gradTensor = savedNNTensors[layer]["grad"].getData();
    // FeatType *dAZ = savedNNTensors[layer]["dAZ"].getData(); // get from AE
    // lambda

    assert(outputTensor != NULL);
    assert(gradTensor != NULL);

#ifdef _GPU_ENABLED_
    printf("Aggregate start back\n");
    // CuMatrix feat;
    // feat.loadSpDense(gradTensor, savedNNTensors[layer]["bg_d"].getData(),
    //                  graph.localVtxCnt, graph.srcGhostCnt, featDim);
    switch (aggregator) {
        case (AGGREGATOR::WSUM): {
            // dA*Z first

            CuMatrix z;
            z.loadSpDense(savedNNTensors[layer]["z"].getData(),
                          savedNNTensors[layer]["fg_z"].getData(),
                          graph.localVtxCnt, graph.srcGhostCnt,
                          savedNNTensors[layer]["z"].getCols());
            CuMatrix dummydA = cu.wrapMatrix(
                savedNNTensors[layer]["dA"]);  // for getting dA into VRAM
            CuMatrix dA = *NormAdjMatrixIn;
            dA.csrVal = dummydA.devPtr;
            CuMatrix dAZ = cu.aggregate(dA, z);

            //A.dP
            CuMatrix dP;
            dP.loadSpDense(gradTensor, savedNNTensors[layer]["bg_d"].getData(),
                           graph.localVtxCnt, graph.dstGhostCnt,
                           savedNNTensors[layer]["bg_d"].getCols());

            CuMatrix dummyA= cu.wrapMatrix(Matrix(1,graph.backwardAdj.nnz,graph.backwardAdj.values));
            CuMatrix A = *NormAdjMatrixOut;
            CuMatrix AdP = cu.aggregate(A, dP);
            cu.hadamardAdd(AdP,dAZ);
            CuMatrix res=AdP;
            res.setData(outputTensor);
            res.updateMatrixFromGPU();
            CuMatrix::freeGPU();
            std::cout << "Finish GPU aggregation\n";
            break;
        }
        default:
            printLog(nodeId, "Invalid aggregator %d", aggregator);
            break;
    }
    currId = graph.localVtxCnt;
#else
    switch (aggregator) {
        case (AGGREGATOR::WSUM): {
            memcpy(outputTensor, gradTensor,
                   sizeof(FeatType) * graph.localVtxCnt * featDim);

            AggOPArgs args = {outputTensor, eVGradTensor, graph.localVtxCnt,
                              edgsCnt, featDim};
            auto computeFn =
                std::bind(&Engine::aggregateBPCompute, this,
                          std::placeholders::_1, std::placeholders::_2);
            computePool->perform(computeFn, &args);
            computePool->sync();

            break;
        }
        default:
            printLog(nodeId, "Invalid aggregator %d", aggregator);
            break;
    }
#endif

    vecTimeAggregate[2 * numLayers - layer - 1] += getTimer() - sttTimer;

    return outputTensor;
}

// Async pipeline
void Engine::aggregator(unsigned tid) {
    BackoffSleeper bs;

    while (!aggHalt) {
        aggQueueLock.lock();
        if (aggregateQueue.empty()) {
            aggQueueLock.unlock();
            bs.sleep();
            continue;
        }

        Chunk c = aggregateQueue.top();
        // check before aggregation
        if (c.isFirstLayer()) {
            // Converge state changes
            if (convergeState != CONVERGE_STATE::EARLY && maxEpoch == 0 &&
                staleness != UINT_MAX) {
                // record the current max epoch of all nodes.
                nodeManager.readEpochUpdates();

                // maxEpoch = minEpoch + staleness;
                maxEpoch = minEpoch + 1;
                unsigned maxe = minEpoch + staleness;
                if (nodesFinishedEpoch[maxe % (staleness + 1)] > 0) {
                    maxEpoch = maxe;
                }
                maxe--;
                while (maxe >= minEpoch) {
                    if (nodesFinishedEpoch[maxe % (staleness + 1)] > 0) {
                        maxEpoch = maxe + 1;
                        break;
                    }
                    maxe--;
                }
                printLog(nodeId, "max epoch %u, curr epoch %u", maxEpoch,
                         currEpoch);
            }
            if (c.epoch == numEpochs ||
                (convergeState != CONVERGE_STATE::EARLY &&
                 c.epoch == maxEpoch + 1)) {
                while (!aggregateQueue.empty()) {
                    aggregateQueue.pop();
                    ++finishedChunks;
                }
                aggQueueLock.unlock();
                if (finishedChunks == numLambdasForward) {
                    if (convergeState != CONVERGE_STATE::EARLY) {
                        extern std::string
                            CONVERGE_STATE_STR[CONVERGE_STATE::NUM_STATE];
                        printLog(nodeId, "STATE changes to %s at epoch %u",
                                 CONVERGE_STATE_STR[convergeState].c_str(),
                                 currEpoch);
                    }
                    ++currEpoch;
                    aggHalt = true;
                    break;
                }

                bs.sleep();
                continue;

                // There is a chunk but it is beyond the staleness bound
            } else if (staleness != UINT_MAX &&
                       c.epoch > minEpoch + staleness) {
                // Read incoming message buffer to see if there are
                // updates to the minEpoch
                nodeManager.readEpochUpdates();
                aggQueueLock.unlock();

                bs.sleep();
                // Messages to check how often a node is blocking
                if ((bs.trails + 1) % 1000 == 0) {
                    printLog(nodeId, "Blocking. MinE %u, Finished %u", minEpoch,
                             nodesFinishedEpoch[minEpoch % (staleness + 1)]);
                }
                continue;
            }
        }

        if (c.isFirstLayer() && c.epoch == currEpoch + 1) {
            ++currEpoch;
            ++numAsyncEpochs;
            printLog(nodeId, "STARTING epoch %u [%u]", currEpoch, c.epoch);
        }

        // printLog(nodeId, "AGGREGATE: Got %s", c.str().c_str());
        aggregateQueue.pop();
        aggQueueLock.unlock();

        double startAgg = getTimer();
        if (c.dir == PROP_TYPE::FORWARD) {
            aggregateChunk(c);
        } else {
            aggregateBPChunk(c);
        }
        vecTimeAggregate[c.dir * numLayers + c.layer] += getTimer() - startAgg;
        resComm->NNCompute(c);

        bs.reset();
    }
}

/////////////////////////////////////////////////////////
// Below are private forward functions for the engine. //
/////////////////////////////////////////////////////////
void Engine::aggregateCompute(unsigned tid, void *args) {
    FeatType *outputTensor = ((AggOPArgs *)args)->outputTensor;
    FeatType **eVFeatsTensor = ((AggOPArgs *)args)->inputTensor;
    const unsigned vtcsCnt = ((AggOPArgs *)args)->vtcsCnt;
    // const unsigned edgsCnt = ((AggOPArgs *) args)->edgsCnt;
    const unsigned featDim = ((AggOPArgs *)args)->featDim;

    unsigned lvid = 0;
    while (currId < vtcsCnt) {
        lvid = __sync_fetch_and_add(&currId, 1);
        if (lvid < vtcsCnt) {
            forwardAggregateFromNeighbors(lvid, outputTensor, eVFeatsTensor,
                                          featDim);
        }
    }
}

void Engine::aggregateChunk(Chunk &c) {
    unsigned lvid = c.lowBound;
    unsigned limit = c.upBound;
    unsigned featDim = getFeatDim(c.layer);

    FeatType *featTensor = NULL;
    if (c.layer == 0)
        featTensor =
            getVtxFeat(savedNNTensors[c.layer]["x"].getData(), lvid, featDim);
    else
        featTensor = getVtxFeat(savedNNTensors[c.layer - 1]["h"].getData(),
                                lvid, featDim);

    FeatType *aggTensor = savedNNTensors[c.layer]["ah"].getData();
    FeatType **eFeatsTensor = savedEdgeTensors[c.layer]["fedge"];

    FeatType *chunkPtr = getVtxFeat(aggTensor, lvid, featDim);
    std::memcpy(chunkPtr, featTensor,
                sizeof(FeatType) * (limit - lvid) * featDim);
    while (lvid < limit) {
        forwardAggregateFromNeighbors(lvid++, aggTensor, eFeatsTensor,
                                      getFeatDim(c.layer));
    }
}

/**
 *
 * Aggregate featDim feature values starting from offset from all neighbors
 * (including self). Then write the results to the data buffer area for
 * serialization. The results are to be used for being sent to lambda threads.
 *
 */
inline void Engine::forwardAggregateFromNeighbors(unsigned lvid,
                                                  FeatType *outputTensor,
                                                  FeatType **inputTensor,
                                                  unsigned featDim) {
    // Read out data of the current layer of given vertex.
    FeatType *currDataDst = getVtxFeat(outputTensor, lvid, featDim);

    // Aggregate from incoming neighbors.
    for (unsigned long long eid = graph.forwardAdj.columnPtrs[lvid];
         eid < graph.forwardAdj.columnPtrs[lvid + 1]; ++eid) {
        EdgeType edgeWeight = graph.forwardAdj.values[eid];
        for (unsigned j = 0; j < featDim; ++j) {
            currDataDst[j] += inputTensor[eid][j] * edgeWeight;
        }
    }
}

//////////////////////////////////////////////////////////
// Below are private backward functions for the engine. //
//////////////////////////////////////////////////////////
void Engine::aggregateBPCompute(unsigned tid, void *args) {
    FeatType *outputTensor = savedNNTensors[layer]["aTg"].getData();

    FeatType **eFeatTensor = savedEdgeTensors[layer]["fedge"];
    FeatType **eGradTensor = savedEdgeTensors[layer]["bedge"];
    FeatType *dA = savedNNTensors[layer]["dA"].getData();

    const unsigned vtcsCnt = ((AggOPArgs *)args)->vtcsCnt;
    // const unsigned edgsCnt = ((AggOPArgs *) args)->edgsCnt;
    const unsigned featDim = ((AggOPArgs *)args)->featDim;
    // printLog(nodeId, "featDim of back agg %u", featDim);

    unsigned lvid = 0;
    while (currId < vtcsCnt) {
        lvid = __sync_fetch_and_add(&currId, 1);
        if (lvid < vtcsCnt) {
            // Read out data of the current layer of given vertex.
            FeatType *currDataDst = getVtxFeat(outputTensor, lvid, featDim);

            // Aggregate from outgoing neighbors.
            // TODO: (YIFAN) Here should be backwardAdj and rowPtrs. Now this
            // works only for undirected graph
            for (unsigned long long eid = graph.backwardAdj.rowPtrs[lvid];
                 eid < graph.backwardAdj.rowPtrs[lvid + 1]; ++eid) {
                EdgeType normFactor = graph.backwardAdj.values[eid];
                for (unsigned j = 0; j < featDim; ++j) {
                    currDataDst[j] += eGradTensor[eid][j] * normFactor;
                }
            }

            // Aggregate from incoming neighbors.
            for (unsigned long long eid = graph.forwardAdj.columnPtrs[lvid];
                 eid < graph.forwardAdj.columnPtrs[lvid + 1]; ++eid) {
                EdgeType edgeGrad = dA[eid];
                for (unsigned j = 0; j < featDim; ++j) {
                    currDataDst[j] += eFeatTensor[eid][j] * edgeGrad;
                }
            }
        }
    }
}

void Engine::aggregateBPChunk(Chunk &c) {
    unsigned lvid = c.lowBound;
    unsigned limit = c.upBound;
    unsigned featDim = getFeatDim(c.layer + 1);

    FeatType *featTensor =
        getVtxFeat(savedNNTensors[c.layer]["grad"].getData(), lvid, featDim);
    FeatType *aggTensor = savedNNTensors[c.layer]["aTg"].getData();
    FeatType **eFeatsTensor = savedEdgeTensors[c.layer]["bedge"];

    FeatType *chunkPtr = getVtxFeat(aggTensor, lvid, featDim);
    std::memcpy(chunkPtr, featTensor,
                sizeof(FeatType) * (limit - lvid) * featDim);
    while (lvid < limit) {
        backwardAggregateFromNeighbors(lvid++, aggTensor, eFeatsTensor,
                                       featDim);
    }
}

/**
 *
 * Aggregate featDim feature values starting from offset from all neighbors
 * (including self). Then write the results to the data buffer area for
 * serialization. The results are to be used for being sent to lambda threads.
 *
 */
void Engine::backwardAggregateFromNeighbors(unsigned lvid,
                                            FeatType *nextGradTensor,
                                            FeatType **gradTensor,
                                            unsigned featDim) {
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
    for (unsigned long long eid = graph.forwardAdj.columnPtrs[lvid];
         eid < graph.forwardAdj.columnPtrs[lvid + 1]; ++eid) {
        EdgeType normFactor = graph.forwardAdj.values[eid];
        for (unsigned j = 0; j < featDim; ++j) {
            currDataDst[j] += gradTensor[eid][j] * normFactor;
        }
    }
}
