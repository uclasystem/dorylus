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


FeatType *Engine::aggregate(FeatType **edgsTensor, unsigned edgsCnt,
                            unsigned featDim, AGGREGATOR aggregator) {
    double sttTimer = getTimer();
#ifdef _GPU_ENABLED_
    std::cout << "Start GPU aggregation forward\n";

    // Loading tensor in CPU
    FeatType *outputTensor = savedNNTensors[layer]["ah"].getData();
    FeatType *gTensor = savedNNTensors[layer]["fg"].getData();
    FeatType *hTensor = (layer == 0) ? savedNNTensors[layer]["x"].getData()
                                     : savedNNTensors[layer - 1]["h"].getData();

    // Load Feature into GPU memory
    CuMatrix feat;
    feat.loadSpDense(hTensor, gTensor, graph.localVtxCnt, graph.srcGhostCnt,
                     featDim);
    switch (aggregator) {
        case (AGGREGATOR::WSUM): {
            CuMatrix out = cu.aggregate(*NormAdjMatrixIn, feat, *Norms);
            out.setData(outputTensor);
            out.updateMatrixFromGPU();
            cudaDeviceSynchronize();

            break;
        }
        default:
            printLog(nodeId, "Invalid Aggregator %d.", aggregator);
            break;
    }
    currId = graph.localVtxCnt;
#else
    // AH
    FeatType* hTensor = savedNNTensors[layer]["z"].getData();
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
    // FeatType *dAZ = savedNNTensors[layer]["dAZ"].getData(); // get from AE lambda

    assert(outputTensor != NULL);
    assert(gradTensor != NULL);

#ifdef _GPU_ENABLED_
    printf("Aggregate start back\n");
    CuMatrix feat;
    feat.loadSpDense(gradTensor, savedNNTensors[layer]["bg_d"].getData(),
                     graph.localVtxCnt, graph.dstGhostCnt, featDim);
    switch (aggregator) {
        case (AGGREGATOR::WSUM): {
            CuMatrix out = cu.aggregate(*NormAdjMatrixOut, feat, *Norms);
            out.setData(outputTensor);
            out.updateMatrixFromGPU();
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
        } else {
            Chunk c = aggregateQueue.top();
            aggregateQueue.pop();
            printLog(nodeId, "Aggregating %s", c.str().c_str());
            sleep_ms(1000);
            aggQueueLock.unlock();

            double startAgg = getTimer();
            if (c.dir == PROP_TYPE::FORWARD) {
                aggregateChunk(c);
            } else {
                aggregateBPChunk(c);
            }

            vecTimeAggregate[c.dir * numLayers + c.layer] +=
                getTimer() - startAgg;

            if (c.dir == PROP_TYPE::FORWARD && c.layer < numLayers - 1) {
                c.layer += 1;
                c.vertex = true;
                resComm->NNCompute(c);
                printLog(nodeId, "Launching AV for Layer %u", c.layer);
            } else if (c.dir == PROP_TYPE::FORWARD) {
                c.vertex = false;
                c.dir = PROP_TYPE::BACKWARD;

                Matrix& labels = savedNNTensors[numLayers - 1]["lab"];
                FeatType* labelPtr = labels.get(c.lowBound);
                FeatType* outputDeriv = savedNNTensors[numLayers - 1]["grad"].get(c.lowBound);
                FeatType* agg = savedNNTensors[numLayers - 1]["az"].get(c.lowBound);


                unsigned rows = c.upBound - c.lowBound;
                unsigned cols = labels.getCols();
                softmax(agg, outputDeriv, rows, cols);

                for (unsigned i = 0; i < rows * cols; ++i) {
                    outputDeriv[i] -= labelPtr[i];
                }

                printLog(nodeId, "Calculated predictions and launched AEB");
                scatQueueLock.lock();
                scatterQueue.push(c);
                scatQueueLock.lock();
            } else {
                resComm->NNCompute(c);
            }

            bs.reset();
        }
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
    unsigned featDim = getFeatDim(c.layer + 1);

    if (nodeId == 0) printLog(nodeId, "FEAT DIM %u", featDim);

    FeatType *featTensor = savedNNTensors[c.layer]["z"].getData();
    FeatType *aggTensor = savedNNTensors[c.layer]["az"].getData();
    FeatType **eFeatsTensor = savedEdgeTensors[c.layer]["fedge"];

    FeatType *chunkPtr = getVtxFeat(aggTensor, lvid, featDim);
    std::memcpy(chunkPtr, featTensor,
                sizeof(FeatType) * (limit - lvid) * featDim);
    while (lvid < limit) {
        forwardAggregateFromNeighbors(lvid++, aggTensor, eFeatsTensor,
                                      getFeatDim(c.layer + 1));
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
            // TODO: (YIFAN) Here should be backwardAdj and rowPtrs. Now this works only for undirected graph
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

    FeatType *featTensor = getVtxFeat(
        savedNNTensors[c.layer]["grad"].getData(), lvid, featDim);
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
