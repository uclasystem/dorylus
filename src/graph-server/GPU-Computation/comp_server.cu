#include <cmath>
#include <iostream>
#include <thread>

#include "../../common/utils.hpp"
#include "comp_server.cuh"
using namespace std;
void loadWeightServers(std::vector<char *> &addresses,
                       const std::string &wServersFile) {
    std::ifstream infile(wServersFile);
    if (!infile.good())
        printf("Cannot open weight server file: %s [Reason: %s]\n",
               wServersFile.c_str(), std::strerror(errno));

    assert(infile.good());

    std::string line;
    while (!infile.eof()) {
        std::getline(infile, line);
        boost::algorithm::trim(line);

        if (line.length() == 0) continue;

        char *addr = strdup(line.c_str());
        addresses.push_back(addr);
    }
}
ComputingServer::ComputingServer() : cu(ComputingUnit::getInstance()){};

ComputingServer::ComputingServer(GPUComm *gpu_comm)
    : cu(ComputingUnit::getInstance()) {
    gpuComm = gpu_comm;
    totalLayers = gpu_comm->totalLayers;
    nodeId = gpu_comm->nodeId;
    msgService = MessageService(gpu_comm->wPort, nodeId);
    loadWeightServers(weightServerAddrs, gpu_comm->wServersFile);
    msgService.setUpWeightSocket(
        weightServerAddrs.at(nodeId % weightServerAddrs.size()));

    msgService.prefetchWeightsMatrix(totalLayers);
}

// Start listening to main thread
// void ComputingServer::terminate() {
//     msgService.terminateWeightServers(weightServerAddrs);
// }

void ComputingServer::processForward(unsigned layer, bool lastLayer) {

    Matrix feats = (*gpuComm->tensorMap)["ah"];
    Matrix weight = msgService.getWeightMatrix(layer);
    CuMatrix z = cu.dot(feats, weight);

    if (!lastLayer) {
        Matrix savedTensor = (*gpuComm->tensorMap)["z"];
        Matrix outputTensor = (*gpuComm->tensorMap)["h"];
        FeatType *act_z = outputTensor.getData();
        FeatType *z_data = savedTensor.getData();
        memcpy(z_data, z.getData(), z.getDataSize());
        cu.activate(z);  // z data get activated ...
        z.updateMatrixFromGPU();
        memcpy(act_z, z.getData(), z.getDataSize());

    } else {  // do the last layer + the bp with it
        CuMatrix cuPredictions = cu.softmaxRows(z);
        gradLoss(layer, cuPredictions);
    }
    delete[] z.getData();
    CuMatrix::freeGPU();
}

void ComputingServer::processBackward(unsigned layer) {
    gradLayer(layer);
    if (layer == 0) msgService.prefetchWeightsMatrix(totalLayers);
}

void ComputingServer::gradLayer(unsigned layer) {
    Matrix grad = (*gpuComm->tensorMap)["aTg"];
    CuMatrix cuGrad = cu.wrapMatrix(grad);
    Matrix z = (*gpuComm->tensorMap)["z"];
    CuMatrix cuZ = cu.wrapMatrix(z);
    Matrix h = (*gpuComm->tensorMap)["h"];
    CuMatrix cuH = cu.wrapMatrix(h);
    Matrix ah = (*gpuComm->tensorMap)["ah"];
    CuMatrix cuAh = cu.wrapMatrix(ah);

    CuMatrix interGrad = cu.activateBackward(cuZ, cuH, cuGrad);
    CuMatrix cuWeightUpdates = cuAh.dot(interGrad, true, false);

    Matrix weightUpdates = cuWeightUpdates.getMatrix();
    Matrix weight = msgService.getWeightMatrix(layer);
    CuMatrix cuWeights = cu.wrapMatrix(weight);
    if (layer != 0) {
        CuMatrix resultGrad = interGrad.dot(cuWeights, false, true);
        resultGrad.setData((*gpuComm->tensorMap)["grad"].getData());
        resultGrad.updateMatrixFromGPU();
    }

    msgService.sendWeightUpdate(weightUpdates, layer);
    CuMatrix::freeGPU();
}

void ComputingServer::gradLoss(unsigned layer, CuMatrix pred, bool report) {
    // here it can be optimized by fetching directly from Forward;
    Matrix labels = (*gpuComm->tensorMap)["lab"];
    CuMatrix cuLabels = cu.wrapMatrix(labels);

    if (report) {
        Matrix cpuPreds = pred.getMatrix();
        // Asynchronously do acc, loss calc on CPUs
        std::thread evalThread([&](Matrix labels) {
            Matrix cpuPreds = pred.getMatrix();
            float acc = 0.0, loss = 0.0;
            unsigned featDim = labels.getCols();
            unsigned valStt = (unsigned)(labels.getRows() * TRAIN_PORTION);
            unsigned valEnd = valStt + (unsigned)(labels.getRows() * VAL_PORTION);
            for (unsigned i = valStt; i < valEnd; i++) {
                FeatType *currLabel = labels.getData() + i * labels.getCols();
                FeatType *currPred = cpuPreds.getData() + i * labels.getCols();
                acc += currLabel[argmax(currPred, currPred + featDim)];
                loss -= std::log(currPred[argmax(currLabel, currLabel + featDim)]);
            }
            printLog(nodeId, "ACC %f, LOSS %f", acc, loss);
            msgService.sendAccloss(acc, loss, cpuPreds.getRows());
            unsigned valsetSize = (unsigned)(cpuPreds.getRows() * VAL_PORTION);
            printLog(nodeId, "batch Acc: %f, Loss: %f", acc / valsetSize, loss / valsetSize);
            cpuPreds.free();
        },
        labels);
        evalThread.detach();
//        float acc, loss;
//        cu.getTrainStat(pred, cuLabels, acc, loss);
//        unsigned valsetSize = (unsigned)(pred.getRows() * VAL_PORTION);
//        msgService.sendAccloss(acc, loss, pred.getRows());
//        // printLog(nodeId, "valset size %u, total size %u", valsetSize, pred.getRows());
//        printLog(nodeId, "batch Acc: %f, Loss: %f", acc / valsetSize, loss / valsetSize);
    }
    cu.maskout(pred, cuLabels);

    CuMatrix d_output = cu.hadamardSub(pred, cuLabels);
    d_output.scale(1.0 / (gpuComm->engine->graph.globalVtxCnt * TRAIN_PORTION));

    Matrix weight = msgService.getWeightMatrix(layer);
    CuMatrix cuWeights = cu.wrapMatrix(weight);
    CuMatrix interGrad = d_output.dot(cuWeights, false, true);
    interGrad.setData((*gpuComm->tensorMap)["grad"].getData());
    interGrad.updateMatrixFromGPU();

    Matrix ah = (*gpuComm->tensorMap)["ah"];
    CuMatrix cuAh = cu.wrapMatrix(ah);
    CuMatrix cuWeightUpdates = cuAh.dot(d_output, true, false);
    Matrix weightUpdates = cuWeightUpdates.getMatrix();
    msgService.sendWeightUpdate(weightUpdates, layer);
}
