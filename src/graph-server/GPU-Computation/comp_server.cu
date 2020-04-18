#include "../../common/utils.hpp"
#include "comp_server.cuh"

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

    Matrix outputTensor = (*gpuComm->tensorMap)["h"];
    Matrix savedTensor = (*gpuComm->tensorMap)["z"];
    Matrix feats = (*gpuComm->tensorMap)["ah"];

    FeatType *z_data = savedTensor.getData();
    FeatType *act_z = outputTensor.getData();

    Matrix weight = msgService.getWeightMatrix(layer);
    CuMatrix z = cu.dot(feats, weight);
    memcpy(z_data, z.getData(), z.getDataSize());
    if (!lastLayer) {
        cu.activate(z);  // z data get activated ...
        z.updateMatrixFromGPU();
        memcpy(act_z, z.getData(), z.getDataSize());
    } else { //do the last layer + the bp with it
        CuMatrix cuPredictions = cu.softmaxRows(z);
        cuPredictions.updateMatrixFromGPU();
        memcpy(act_z, cuPredictions.getData(), z.getDataSize());
        delete[] cuPredictions.getData();
        gradLoss(layer);
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
    CuMatrix resultGrad = interGrad.dot(cuWeights, false, true);
    resultGrad.setData((*gpuComm->tensorMap)["grad"].getData());
    resultGrad.updateMatrixFromGPU();
    Matrix ah = (*gpuComm->tensorMap)["ah"];
    CuMatrix cuAh = cu.wrapMatrix(ah);
    CuMatrix cuWeightUpdates = cuAh.dot(interGrad, true, false);
    Matrix weightUpdates = cuWeightUpdates.getMatrix();

    msgService.sendWeightUpdate(weightUpdates, layer);
}

void ComputingServer::gradLoss(unsigned layer) {
    // here it can be optimized by fetching directly from Forward;
    Matrix predictions = (*gpuComm->tensorMap)["h"];
    Matrix labels = (*gpuComm->tensorMap)["lab"];

    CuMatrix cuPredictions = cu.wrapMatrix(predictions);
    CuMatrix cuLabels = cu.wrapMatrix(labels);

    CuMatrix d_output = cu.hadamardSub(cuPredictions, cuLabels);
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

// void ComputingServer::evaluateModel(Matrix& activations){
//     CuMatrix labels = cu.wrapMatrix(msgService.requestMatrix());
//     CuMatrix cuAct =cu.wrapMatrix(activations);
//     CuMatrix cuPredictions = cu.softmaxRows(cuAct);

//     // Check if the label with the highest probability after softmax is equal
//     to the
//     // target label
//     unsigned totalCorrect = cu.checkAccuracy(cuPredictions, labels);

//     // Sum the individual losses of each vertex for this validation partition
//     float lossThisPart = cu.checkLoss(cuPredictions, labels);
// }
