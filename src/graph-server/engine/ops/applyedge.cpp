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


FeatType **Engine::applyEdge(EdgeType *edgsTensor, unsigned edgsCnt,
                             unsigned eFeatDim, FeatType **eSrcVtcsTensor,
                             FeatType **eDstVtcsTensor, unsigned inFeatDim,
                             unsigned outFeatDim) {
    double sttTimer = getTimer();

    // do nothing
    FeatType **outputTensor = eSrcVtcsTensor;
    eSrcVtcsTensor = NULL;

    unsigned vtcsCnt = graph.localVtxCnt;
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
                currEpoch,     true};  // epoch is not useful in sync version
            resComm->NNCompute(chunk);

            availLambdaId++;
        }
        vecTimeLambdaInvoke[layer] += getTimer() - invTimer;

        double waitTimer = getTimer();
        resComm->NNSync();
        vecTimeLambdaWait[layer] += getTimer() - waitTimer;

        // // NOTE: This should definitely be in resComm->NNComute() but since we are
        // // doing everything locally and currently have no infrastructre for switching
        // // b/w tensor on lambdas and CPUs I'm just going to do it here
        // FeatType* az_i = savedNNTensors[layer]["az_i"].getData();
        // FeatType* az_j = savedNNTensors[layer]["az_j"].getData();
        // FeatType* fg_az = savedNNTensors[layer]["fg_az"].getData();

        // FeatType* lRlu = savedNNTensors[layer]["lRlu"].getData();
        // FeatType* eij = savedNNTensors[layer]["eij"].getData();

        // // Compute unnormalized attention scores for all edges in this partition
        // CSCMatrix<EdgeType>& csc = graph.forwardAdj;
        // for (unsigned lvid = 0; lvid < csc.columnCnt; ++lvid) {
        //     FeatType az_lvid = az_i[lvid];
        //     for (unsigned long long eid = csc.columnPtrs[lvid];
        //     eid < csc.columnPtrs[lvid + 1]; ++eid) {
        //         unsigned rowId = graph.forwardAdj.rowIdxs[eid];
        //         FeatType az_rvid = 0.0;
        //         if (rowId < graph.localVtxCnt) {
        //             az_rvid = az_j[rowId];
        //         } else {
        //             az_rvid = fg_az[rowId - graph.localVtxCnt];
        //         }
        //         // az_rvid should never be 0
        //         assert(az_rvid != 0.0);

        //         lRlu[eid] = az_lvid + az_rvid;
        //         eij[eid] = leakyReLU(lRlu[eid]);
        //     }
        // }
    } else {
        Chunk batch{
            nodeId,    nodeId, 0, vtcsCnt, layer, PROP_TYPE::FORWARD,
            currEpoch, true};  // for now it loads the entire feature matrix
        resComm->NNCompute(batch);
    }

    vecTimeApplyEdg[layer] += getTimer() - sttTimer;
    return outputTensor;
}


FeatType **Engine::applyEdgeBackward(FeatType* edgsTensor, unsigned edgsCnt,
                                     unsigned eFeatDim,
                                     FeatType **eSrcVGradTensor,
                                     FeatType **eDstVGradTensor,
                                     unsigned inFeatDim, unsigned outFeatDim) {
    double sttTimer = getTimer();

    FeatType **outputTensor = eDstVGradTensor;
    eDstVGradTensor = NULL;

    if (mode == LAMBDA) {
        double invTimer = getTimer();
        const unsigned chunkSize =
            (graph.localVtxCnt + numLambdasForward - 1) / numLambdasForward;
        unsigned availLambdaId = 0;
        while (availLambdaId < numLambdasForward) {
            unsigned lowBound = availLambdaId * chunkSize;
            unsigned upBound = std::min(lowBound + chunkSize, graph.localVtxCnt);
            Chunk chunk{
                availLambdaId,  nodeId * numLambdasForward + availLambdaId,
                lowBound,       upBound,
                layer,          PROP_TYPE::BACKWARD,
                currEpoch,      false};
            resComm->NNCompute(chunk);

            ++availLambdaId;
        }
        vecTimeLambdaInvoke[2 * numLayers - layer - 1] += getTimer() - invTimer;

        double waitTimer = getTimer();
        resComm->NNSync();
        vecTimeLambdaWait[2 * numLayers - layer - 1] += getTimer() - waitTimer;
    }

//    Matrix& derivMat = savedNNTensors[layer + 1]["grad"];
//    Matrix& Z = savedNNTensors[layer]["z"];
//    Matrix& eij = savedNNTensors[layer]["eij"];
//    //Matrix& lRlu = savedNNTensors[layer]["lRlu"];
//
//    CSCMatrix<EdgeType>& csc = graph.forwardAdj;
//
//    // derivMat \dot Z^T
//    printLog(nodeId, "%s    %s", derivMat.shape().c_str(),
//             Z.shape().c_str());
//    Matrix d_ae = derivMat.dot(Z, false, true);
//
//    Matrix d_sm = softmax_prime(eij.getData(), csc.values, csc.nnz);
//
//    printLog(nodeId, "SM_PRIME * DAE: %s    %s",
//      d_ae.shape().c_str(), d_sm.shape().c_str());
//
//    Matrix d_2 = d_ae * d_sm;

    // NOTE: Don't forget to clean up tensors after this phase

//    for (auto &sTensor : edgNNSavedTensors[layer - 1]) {
//        delete[] sTensor.getData();
//    }
    vecTimeApplyEdg[2 * numLayers - layer - 1] += getTimer() - sttTimer;
    return outputTensor;
}



/////////////////////////////////////////////////////////
// Below are private forward functions for the engine. //
/////////////////////////////////////////////////////////


//////////////////////////////////////////////////////////
// Below are private backward functions for the engine. //
//////////////////////////////////////////////////////////


// reshape vtcs tensor to edgs tensor. Each element in edgsTensor is a reference
// to a vertex feature. Both src vtx features and dst vtx features included in
// edgsTensor. [srcV Feats (local inEdge cnt); dstV Feats (local inEdge cnt)]
FeatType **Engine::srcVFeats2eFeats(FeatType *vtcsTensor, FeatType *ghostTensor,
                                    unsigned vtcsCnt, unsigned featDim) {
    underlyingVtcsTensorBuf = vtcsTensor;

    FeatType **eVtxFeatsBuf = new FeatType *[2 * graph.localInEdgeCnt];
    FeatType **eSrcVtxFeats = eVtxFeatsBuf;
    FeatType **eDstVtxFeats = eSrcVtxFeats + graph.localInEdgeCnt;

    unsigned long long edgeItr = 0;
    for (unsigned lvid = 0; lvid < graph.localVtxCnt; ++lvid) {
        for (unsigned long long eid = graph.forwardAdj.columnPtrs[lvid];
             eid < graph.forwardAdj.columnPtrs[lvid + 1]; ++eid) {
            unsigned srcVid = graph.forwardAdj.rowIdxs[eid];
            if (srcVid < graph.localVtxCnt) {
                eSrcVtxFeats[edgeItr] = getVtxFeat(vtcsTensor, srcVid, featDim);
            } else {
                eSrcVtxFeats[edgeItr] = getVtxFeat(
                    ghostTensor, srcVid - graph.localVtxCnt, featDim);
            }
            eDstVtxFeats[edgeItr] = getVtxFeat(vtcsTensor, lvid, featDim);
            ++edgeItr;
        }
    }

    return eVtxFeatsBuf;
}

// similar to srcVFeats2eFeats, but based on outEdges of local vertices.
// [dstV Feats (local outEdge cnt); srcV Feats (local outEdge cnt)]
FeatType **Engine::dstVFeats2eFeats(FeatType *vtcsTensor, FeatType *ghostTensor,
                                    unsigned vtcsCnt, unsigned featDim) {
    underlyingVtcsTensorBuf = vtcsTensor;

    FeatType **eVtxFeatsBuf = new FeatType *[2 * graph.localOutEdgeCnt];
    FeatType **eSrcVtxFeats = eVtxFeatsBuf;
    FeatType **eDstVtxFeats = eSrcVtxFeats + graph.localOutEdgeCnt;

    unsigned long long edgeItr = 0;
    for (unsigned lvid = 0; lvid < graph.localVtxCnt; ++lvid) {
        for (unsigned long long eid = graph.backwardAdj.rowPtrs[lvid];
             eid < graph.backwardAdj.rowPtrs[lvid + 1]; ++eid) {
            unsigned srcVid = graph.backwardAdj.columnIdxs[eid];
            if (srcVid < graph.localVtxCnt) {
                eSrcVtxFeats[edgeItr] = getVtxFeat(vtcsTensor, srcVid, featDim);
            } else {
                eSrcVtxFeats[edgeItr] = getVtxFeat(
                    ghostTensor, srcVid - graph.localVtxCnt, featDim);
            }
            eDstVtxFeats[edgeItr] = getVtxFeat(vtcsTensor, lvid, featDim);
            ++edgeItr;
        }
    }

    return eVtxFeatsBuf;
}
