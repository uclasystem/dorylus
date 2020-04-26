#include <iostream>

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

    // send INFO to weight server
    unsigned numNodes = gpuComm->numNodes;
    if (nodeId < weightServerAddrs.size()) {
        unsigned count = 0;
        for (size_t i = 0; i < numNodes; ++i) {
            if (i % weightServerAddrs.size() == nodeId) count += 1;
        }
        msgService.sendInfoMessage(count);
    }
    msgService.prefetchWeightsMatrix(totalLayers);
}

// Start listening to main thread
void ComputingServer::terminate() {
    msgService.terminateWeightServers(weightServerAddrs);
}

void ComputingServer::processForward(unsigned layer, bool lastLayer) {
    if (layer == 0) CuMatrix::freeGPU();

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
    CuMatrix interGrad = cu.activateBackward(cuZ, cuGrad);
    Matrix weight = msgService.getWeightMatrix(layer);
    CuMatrix cuWeights = cu.wrapMatrix(weight);
    if (layer != 0) {
        CuMatrix resultGrad = interGrad.dot(cuWeights, false, true);
        resultGrad.setData((*gpuComm->tensorMap)["grad"].getData());
        resultGrad.updateMatrixFromGPU();
    }

    Matrix ah = (*gpuComm->tensorMap)["ah"];
    CuMatrix cuAh = cu.wrapMatrix(ah);
    CuMatrix cuWeightUpdates = cuAh.dot(interGrad, true, false);
    Matrix weightUpdates = cuWeightUpdates.getMatrix();

    msgService.sendWeightUpdate(weightUpdates, layer);
}

void ComputingServer::gradLoss(unsigned layer, CuMatrix pred, bool report) {
    // here it can be optimized by fetching directly from Forward;
    Matrix labels = (*gpuComm->tensorMap)["lab"];
    CuMatrix cuLabels = cu.wrapMatrix(labels);
    CuMatrix d_output = cu.hadamardSub(pred, cuLabels);

    if (report) {
        float acc, loss;
        cu.getTrainStat(pred, cuLabels, acc, loss);
        printLog(nodeId, "batch Acc: %f, Loss: %f\n", acc, loss);
    }

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
