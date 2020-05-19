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


// FeatType *Engine::fusedGAS(FeatType *vtcsTensor, unsigned vtcsCnt,
//                            unsigned inFeatDim, unsigned outFeatDim,
//                            bool scatter) {
//     double sttTimer = getTimer();
//     // Check just to make sure partition ranges are empty
//     // before starting
//     consumerQueueLock.lock();
//     while (!rangesToScatter.empty()) rangesToScatter.pop();
//     consumerQueueLock.unlock();

//     // Start data receivers
//     commHalt = false;
//     // Forward declaration to ensure pointer remains in scope
//     FeatType *outputTensor = new FeatType[vtcsCnt * outFeatDim];
//     FeatType *zTensor = new FeatType[vtcsCnt * outFeatDim];
//     auto fgr_fp = std::bind(&Engine::forwardGhostReceiver, this,
//     std::placeholders::_1); auto fgu_fp =
//     std::bind(&Engine::pipelineForwardGhostUpdates, this,
//                     std::placeholders::_1, std::placeholders::_2);
//     std::thread scatterThread;
//     if (scatter) {
//         forwardGhostVerticesDataOut = new FeatType[graph.srcGhostCnt *
//         outFeatDim]; dataPool->perform(fgr_fp); scatterThread =
//         std::thread(fgu_fp, outputTensor, outFeatDim);
//     }

//     // Prepare for gather phase
//     FeatType *gatheredTensor = new FeatType[vtcsCnt * inFeatDim];
//     currId = 0;
//     AggOPArgs args = {gatheredTensor, vtcsTensor, vtcsCnt, inFeatDim};
//     auto computeFn = std::bind(&Engine::aggregateCompute, this,
//     std::placeholders::_1, std::placeholders::_2);

//     // Start gathering
//     computePool->perform(computeFn, &args);

//     // Prepare for applyVertex phase
//     bool saveInput = true;
//     if (saveInput) {
//         vtxNNSavedTensors[layer].push_back(Matrix(vtcsCnt, inFeatDim,
//         gatheredTensor));
//     }
//     vtxNNSavedTensors[layer].push_back(Matrix(vtcsCnt, outFeatDim, zTensor));
//     Matrix inputTensor_ = Matrix(vtcsCnt, inFeatDim, gatheredTensor);
//     Matrix outputTensor_ = Matrix(vtcsCnt, outFeatDim, outputTensor);
//     resComm->newContext(layer, inputTensor_, outputTensor_,
//     vtxNNSavedTensors, scatter);

//     // Start applyVertex phase
//     unsigned currLambdaId = 0;
//     if (mode == LAMBDA) {
//         const unsigned lambdaChunkSize = (vtcsCnt + numLambdasForward - 1) /
//         numLambdasForward; unsigned availChunkSize = lambdaChunkSize; while
//         (currId < vtcsCnt) {
//             unsigned lvid = currId;
//             while (lvid > availChunkSize) {
//                 resComm->applyVertexForward(layer, currLambdaId, layer ==
//                 numLayers - 1);
//                 ++currLambdaId;
//                 availChunkSize += lambdaChunkSize;
//             }
//             usleep(5000); // wait 5ms and then check again
//         }
//     }
//     computePool->sync();
//     if (mode != LAMBDA) {
//         resComm->requestForward(layer, layer == numLayers - 1);
//     } else {
//         while (currLambdaId < numLambdasForward) {
//             resComm->applyVertexForward(layer, currLambdaId, layer ==
//             numLayers - 1);
//             ++currLambdaId;
//         }
//         resComm->waitResForward(layer, layer == numLayers - 1);
//     }

//     // Wait for all remote schedulings sent by me to be handled.
//     if (scatter) {
//         scatterThread.join();
//     }
//     nodeManager.barrier();
//     commHalt = true;
//     dataPool->sync();

//     // Post-processing for applyVertex phase & clean up
//     bool saveOutput = true;
//     if (saveOutput) {
//         FeatType *outTensorCpy = new FeatType[vtcsCnt * outFeatDim];
//         memcpy(outTensorCpy, outputTensor, vtcsCnt * outFeatDim *
//         sizeof(FeatType)); vtxNNSavedTensors[layer].push_back(Matrix(vtcsCnt,
//         outFeatDim, outTensorCpy));
//     }
//     if (saveInput) {
//         gatheredTensor = NULL;
//     } else {
//         delete[] gatheredTensor;
//     }

//     // Clean up the gather phase
//     if (layer > 0) {
//         delete[] forwardGhostVerticesDataIn;
//         delete[] vtcsTensor;
//     }

//     if (vecTimeAggregate.size() < numLayers) {
//         vecTimeAggregate.push_back(getTimer() - sttTimer);
//     } else {
//         vecTimeAggregate[layer] += getTimer() - sttTimer;
//     }

//     // Set the scattered output as the input for next aggregation phase
//     forwardGhostVerticesDataIn = forwardGhostVerticesDataOut;

//     return outputTensor;
// }


// FeatType *Engine::fusedGASBackward(FeatType *gradTensor, unsigned vtcsCnt,
//                                    unsigned inFeatDim, unsigned outFeatDim,
//                                    bool aggregate, bool scatter) {
//     double sttTimer = getTimer();

//     consumerQueueLock.lock();
//     while (!rangesToScatter.empty()) rangesToScatter.pop();
//     consumerQueueLock.unlock();

//     // Case 1 - First phase, no aggregate needed
//     FeatType* outputTensor = nullptr;
//     if (!aggregate && scatter) {
//         outputTensor = applyScatterPhase(gradTensor, vtcsCnt, inFeatDim,
//         outFeatDim, scatter);
//     }
//     // Case 2 - Full phase including gather, apply, and scatter
//     else if (aggregate && scatter) {
//         outputTensor = aggregateApplyScatterPhase(gradTensor, vtcsCnt,
//         inFeatDim, outFeatDim, scatter);
//     }
//     // Case 3 - Final phase, no scatter needed
//     else if (aggregate && !scatter) {
//         outputTensor = aggregateApplyPhase(gradTensor, vtcsCnt, inFeatDim,
//         outFeatDim, scatter);
//     }
//     else {
//         printLog(nodeId, "\033[1;33m[ UNKOWN ]\033[0m No scatter or
//         aggregate phase");
//     }

//     if (vecTimeAggregate.size() < 2 * numLayers) {
//         for (unsigned i = vecTimeAggregate.size(); i < 2 * numLayers; i++)
//         {
//             vecTimeAggregate.push_back(0.0);
//         }
//     }
//     vecTimeAggregate[numLayers + layer - 1] += getTimer() - sttTimer;

//     backwardGhostVerticesDataIn = backwardGhostVerticesDataOut;

//     return outputTensor;
// }

// Backward scatter phase functions
// FeatType *Engine::applyScatterPhase(FeatType *gradTensor, unsigned vtcsCnt,
//                                     unsigned inFeatDim, unsigned outFeatDim,
//                                     bool scatter) {
//     double sttTimer = getTimer();

//     assert(vtcsCnt == graph.localVtxCnt);
//     commHalt = false;
//     bkwdRecvCnt = 0;

//     FeatType *outputTensor = new FeatType[vtcsCnt * inFeatDim];
//     auto bgr_fp = std::bind(&Engine::backwardGhostReceiver, this,
//                     std::placeholders::_1, std::placeholders::_2);
//     auto bgu_fp = std::bind(&Engine::pipelineBackwardGhostGradients, this,
//                     std::placeholders::_1, std::placeholders::_2);
//     std::thread scatterThread;
//     if (scatter) {
//         backwardGhostVerticesDataOut = new FeatType[graph.dstGhostCnt *
//         inFeatDim]; dataPool->perform(bgr_fp, (void*) &inFeatDim);
//         scatterThread = std::thread(bgu_fp, outputTensor, inFeatDim);
//     }

//     Matrix inputTensor_ = Matrix(vtcsCnt, outFeatDim, gradTensor);
//     Matrix outputTensor_ = Matrix(vtcsCnt, inFeatDim, outputTensor);
//     Matrix targetTensor_ = Matrix(vtcsCnt, getFeatDim(numLayers),
//     localVerticesLabels); resComm->newContext(iteration - 1, inputTensor_,
//     outputTensor_, targetTensor_,
//                         vtxNNSavedTensors, scatter);
//     resComm->requestBackward(iteration - 1, iteration - 1 == numLayers - 1);

//     if (scatter) {
//         scatterThread.join();
//     }
//     commHalt = true;
//     dataPool->sync();

//     delete[] gradTensor;
//     for (auto &sTensor : vtxNNSavedTensors[iteration - 1]) {
//         delete[] sTensor.getData();
//     }
//     vtxNNSavedTensors[iteration - 1].clear();

//     if (vecTimeApplyVtx.size() < 2 * numLayers) {
//         for (unsigned i = vecTimeApplyVtx.size(); i < 2 * numLayers; i++) {
//             vecTimeApplyVtx.push_back(0.0);
//         }
//     }
//     vecTimeApplyVtx[numLayers + iteration - 1] += getTimer() - sttTimer;

//     return outputTensor;
// }

// FeatType *Engine::aggregateApplyScatterPhase(FeatType *gradTensor,
//                                              unsigned vtcsCnt,
//                                              unsigned inFeatDim,
//                                              unsigned outFeatDim,
//                                              bool scatter) {
//     // Prepare for gather phase
//     FeatType *gatheredTensor = new FeatType[vtcsCnt * outFeatDim];
//     FeatType *outputTensor = new FeatType[vtcsCnt * inFeatDim];
//     auto bgr_fp = std::bind(&Engine::backwardGhostReceiver, this,
//                     std::placeholders::_1, std::placeholders::_2);
//     auto bgu_fp = std::bind(&Engine::pipelineBackwardGhostGradients, this,
//                     std::placeholders::_1, std::placeholders::_2);
//     commHalt = false;
//     bkwdRecvCnt = 0;
//     std::thread scatterThread;
//     if (scatter) {
//         backwardGhostVerticesDataOut = new FeatType[graph.dstGhostCnt *
//         outFeatDim]; dataPool->perform(bgr_fp, (void*) &inFeatDim);
//         scatterThread = std::thread(bgu_fp, outputTensor, outFeatDim);
//     }

//     currId = 0;

//     // Start gathering
//     AggOPArgs args = {gatheredTensor, gradTensor, vtcsCnt, outFeatDim};
//     auto computeFn = std::bind(&Engine::aggregateBPCompute, this,
//     std::placeholders::_1, std::placeholders::_2);
//     computePool->perform(computeFn, &args);

//     // Prepare for applyVertex phase
//     Matrix inputTensor_ = Matrix(vtcsCnt, outFeatDim, gatheredTensor);
//     Matrix outputTensor_ = Matrix(vtcsCnt, inFeatDim, outputTensor);
//     Matrix targetTensor_ = Matrix(vtcsCnt, getFeatDim(numLayers),
//     localVerticesLabels); resComm->newContext(layer - 1, inputTensor_,
//     outputTensor_, targetTensor_,
//                         vtxNNSavedTensors, scatter);

//     // Start applyVertex phase
//     unsigned currLambdaId = 0;
//     if (mode == LAMBDA) {
//         const unsigned lambdaChunkSize = (vtcsCnt + numLambdasForward - 1) /
//         numLambdasBackward; unsigned availChunkSize = lambdaChunkSize; while
//         (currId < vtcsCnt) {
//             unsigned lvid = currId;
//             if (lvid > availChunkSize) {
//                 resComm->applyVertexBackward(layer - 1, currLambdaId, layer -
//                 1 == numLayers - 1); availChunkSize += lambdaChunkSize;
//                 ++currLambdaId;
//             }
//             usleep(2000); // wait for 2ms and check again
//         }
//     }
//     computePool->sync();
//     if (mode != LAMBDA) {
//         resComm->requestBackward(layer - 1, layer - 1 == numLayers - 1);
//     } else {
//         while (currLambdaId < numLambdasBackward) {
//             resComm->applyVertexBackward(layer - 1, currLambdaId, layer - 1
//             == numLayers - 1);
//             ++currLambdaId;
//         }
//         resComm->waitResBackward(layer - 1, layer - 1 == numLayers - 1);
//     }

//     if (scatter) {
//         scatterThread.join();
//     }
//     commHalt = true;
//     dataPool->sync();

//     // Clean up applyVertex phase
//     delete[] gatheredTensor;
//     for (auto &sTensor : vtxNNSavedTensors[layer - 1]) {
//         delete[] sTensor.getData();
//     }
//     vtxNNSavedTensors[layer - 1].clear();

//     // Clean up gather phase
//     delete[] gradTensor;
//     delete[] backwardGhostVerticesDataIn;

//     return outputTensor;
// }

// FeatType *Engine::aggregateApplyPhase(FeatType *gradTensor, unsigned vtcsCnt,
//                                       unsigned inFeatDim, unsigned outFeatDim,
//                                       bool scatter) {
//     double sttTimer = getTimer();

//     // Prepare for gather phase
//     FeatType *gatheredTensor = new FeatType[vtcsCnt * outFeatDim];
//     FeatType *outputTensor = new FeatType[vtcsCnt * inFeatDim];
//     currId = 0;

//     // Start gathering
//     AggOPArgs args = {gatheredTensor, gradTensor, vtcsCnt, outFeatDim};
//     auto computeFn = std::bind(&Engine::aggregateBPCompute, this,
//     std::placeholders::_1, std::placeholders::_2);
//     computePool->perform(computeFn, &args);

//     // Prepare for applyVertex phase
//     Matrix inputTensor_ = Matrix(vtcsCnt, outFeatDim, gatheredTensor);
//     Matrix outputTensor_ = Matrix(vtcsCnt, inFeatDim, outputTensor);
//     Matrix targetTensor_ = Matrix(vtcsCnt, getFeatDim(numLayers),
//     localVerticesLabels); resComm->newContext(layer - 1, inputTensor_,
//     outputTensor_, targetTensor_,
//                         vtxNNSavedTensors, scatter);

//     // Start applyVertex phase
//     unsigned currLambdaId = 0;
//     if (mode == LAMBDA) {
//         const unsigned lambdaChunkSize = (vtcsCnt + numLambdasForward - 1) /
//         numLambdasBackward; unsigned availChunkSize = lambdaChunkSize; while
//         (currId < vtcsCnt) {
//             unsigned lvid = currId;
//             if (lvid > availChunkSize) {
//                 resComm->applyVertexBackward(layer - 1, currLambdaId, layer -
//                 1 == numLayers - 1); availChunkSize += lambdaChunkSize;
//                 ++currLambdaId;
//             }
//             usleep(2000); // wait for 2ms and check again
//         }
//     }
//     computePool->sync();
//     if (mode != LAMBDA) {
//         resComm->requestBackward(layer - 1, layer - 1 == numLayers - 1);
//     } else {
//         while (currLambdaId < numLambdasBackward) {
//             resComm->applyVertexBackward(layer - 1, currLambdaId, layer - 1
//             == numLayers - 1);
//             ++currLambdaId;
//         }
//         resComm->waitResBackward(layer - 1, layer - 1 == numLayers - 1);
//     }

//     // Clean up applyVertex phase
//     delete[] gatheredTensor;
//     for (auto &sTensor : vtxNNSavedTensors[layer - 1]) {
//         delete[] sTensor.getData();
//     }
//     vtxNNSavedTensors[layer - 1].clear();

//     // Clean up gather phase
//     delete[] gradTensor;
//     delete[] backwardGhostVerticesDataIn;

//     if (vecTimeAggregate.size() < 2 * numLayers) {
//         for (unsigned i = vecTimeAggregate.size(); i < 2 * numLayers; i++) {
//             vecTimeAggregate.push_back(0.0);
//         }
//     }
//     vecTimeAggregate[numLayers + layer] += getTimer() - sttTimer;

//     return outputTensor;
// }
