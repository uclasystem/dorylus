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


FeatType *Engine::applyVertex(FeatType *vtcsTensor, unsigned vtcsCnt,
                              unsigned inFeatDim, unsigned outFeatDim,
                              bool lastLayer) {
    // Weight fetch. Now we do prefetch for CPU/GPU in their own comm
    // For CPU/GPU only. For lambda this is a void function
    // if (layer == 0) {
    //     resComm->prefetchWeights();
    // }

    double sttTimer = getTimer();
    assert(vtcsCnt == graph.localVtxCnt);
    FeatType *outputTensor = NULL;
    if (lastLayer) {
        outputTensor = savedNNTensors[layer]["grad"].getData();
    } else {
        outputTensor = savedNNTensors[layer]["h"].getData();
    }

    // Start a new lambda communication context.
    if (mode == LAMBDA) {
        double invTimer = getTimer();
        const unsigned chunkSize =
            (vtcsCnt + numLambdasForward - 1) / numLambdasForward;
        unsigned availLambdaId = 0;
        while (availLambdaId < numLambdasForward) {
            unsigned lowBound = availLambdaId * chunkSize;
            unsigned upBound = std::min(lowBound + chunkSize, vtcsCnt);
            Chunk chunk{
                availLambdaId, nodeId * numLambdasForward + availLambdaId,
                lowBound,      upBound,
                layer,         PROP_TYPE::FORWARD,
                currEpoch,     true};
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
    else {
        Chunk batch{
            nodeId,    nodeId, 0, vtcsCnt, layer, PROP_TYPE::FORWARD,
            currEpoch, true};  // for now it loads the entire feature matrix
        resComm->NNCompute(batch);
    }

    if (vecTimeApplyVtx.size() < numLayers) {
        vecTimeApplyVtx.push_back(getTimer() - sttTimer);
    } else {
        vecTimeApplyVtx[layer] += getTimer() - sttTimer;
    }

    return outputTensor;
}

FeatType *Engine::applyVertexBackward(FeatType *gradTensor, unsigned vtcsCnt,
                                      unsigned inFeatDim, unsigned outFeatDim) {
    double sttTimer = getTimer();

    assert(vtcsCnt == graph.localVtxCnt);
    FeatType *outputTensor = savedNNTensors[layer - 1]["grad"].getData();

    if (vecTimeLambdaInvoke.size() < 2 * numLayers) {
        for (unsigned i = vecTimeLambdaInvoke.size(); i < 2 * numLayers; ++i) {
            vecTimeLambdaInvoke.push_back(0.0);
            vecTimeLambdaWait.push_back(0.0);
        }
    }

    if (mode == LAMBDA) {
        for (unsigned u = 0; u < numLambdasForward; ++u) {
            unsigned chunkSize =
                (vtcsCnt + numLambdasForward - 1) / numLambdasForward;
            unsigned lowBound = u * chunkSize;
            unsigned upBound = std::min(lowBound + chunkSize, vtcsCnt);
            Chunk chunk{
                u,         nodeId * numLambdasForward + u,
                lowBound,  upBound,
                layer - 1, PROP_TYPE::BACKWARD,
                currEpoch, true};
            resComm->NNCompute(chunk);
        }
        resComm->NNSync();
    } else {
        Chunk chunk{nodeId,    nodeId,    0,
                    vtcsCnt,   layer - 1, PROP_TYPE::BACKWARD,
                    currEpoch, true};
        resComm->NNCompute(chunk);
    }

    if (vecTimeApplyVtx.size() < 2 * numLayers) {
        for (unsigned i = vecTimeApplyVtx.size(); i < 2 * numLayers; i++) {
            vecTimeApplyVtx.push_back(0.0);
        }
    }
    vecTimeApplyVtx[numLayers + layer - 1] += getTimer() - sttTimer;
    return outputTensor;
}


/////////////////////////////////////////////////////////
// Below are private forward functions for the engine. //
/////////////////////////////////////////////////////////


//////////////////////////////////////////////////////////
// Below are private backward functions for the engine. //
//////////////////////////////////////////////////////////
