#include "comp_server.cuh"
#include "../../common/utils.hpp"

void loadWeightServers(std::vector<char *> &addresses, const std::string &wServersFile) {
    std::ifstream infile(wServersFile);
    if (!infile.good())
        printf("Cannot open weight server file: %s [Reason: %s]\n", wServersFile.c_str(), std::strerror(errno));

    assert(infile.good());

    std::string line;
    while (!infile.eof()) {
        std::getline(infile, line);
        boost::algorithm::trim(line);

        if (line.length() == 0)
            continue;

        char *addr = strdup(line.c_str());
        addresses.push_back(addr);
    }
}
ComputingServer::ComputingServer(): cu(ComputingUnit::getInstance()) {};

ComputingServer::ComputingServer(GPUComm *gpu_comm): cu(ComputingUnit::getInstance()) {
    gpuComm = gpu_comm;
    totalLayers = gpu_comm->totalLayers;
    nodeId = gpu_comm->nodeId;
    msgService = MessageService(gpu_comm->wPort, nodeId);
    loadWeightServers(weightServerAddrs, gpu_comm->wServersFile);
    msgService.setUpWeightSocket(weightServerAddrs.at(nodeId % weightServerAddrs.size()));

    //send INFO to weight server
    unsigned numNodes = gpuComm->numNodes;
    if (nodeId < weightServerAddrs.size()) {
        unsigned count = 0;
        for (size_t i = 0; i < numNodes; ++i) {
            if (i % weightServerAddrs.size() == nodeId)
                count += 1;
        }
        msgService.sendInfoMessage(count);
    }
    msgService.prefetchWeightsMatrix(totalLayers);
}

//Start listening to main thread
void ComputingServer::terminate() {
    msgService.terminateWeightServers(weightServerAddrs);
}

void ComputingServer::processForward(unsigned layer, bool lastLayer) {
    if (layer == 0)
        CuMatrix::freeGPU();
    Matrix feats = gpuComm->inputTensor;
    // TODO: (YIFAN) now we only save Z data as intermediate tensors. So I hard-code 0 here. Need generalize this.
    FeatType *z_data = gpuComm->savedTensors[layer][0].getData();
    FeatType *act_z = gpuComm->outputTensor.getData();

    auto tw = gtimers.getTimer("WeightFetchWait");
    auto tz = gtimers.getTimer("CopyZ");
    auto ta = gtimers.getTimer("CopyA");
    auto tc = gtimers.getTimer("ComputeForward");
    tc->start();
    tw->start();
    Matrix weight = msgService.getWeightMatrix(layer);
    tw->stop();
    CuMatrix z = cu.dot(feats, weight);
    tz->start();
    memcpy(z_data, z.getData(), z.getDataSize());
    tz->stop();
    if (!lastLayer) {
        cu.activate(z);//z data get activated ...
        ta->start();
        z.updateMatrixFromGPU();
        memcpy(act_z, z.getData(), z.getDataSize());
        ta->stop();
    } else {
        CuMatrix cuPredictions = cu.softmaxRows(z);
        ta->start();
        cuPredictions.updateMatrixFromGPU();
        memcpy(act_z, cuPredictions.getData(), z.getDataSize());
        ta->stop();
        delete[] cuPredictions.getData();
    }
    tc->stop();
    delete[] z.getData();
}

// void ComputingServer::evaluateModel(Matrix& activations){
//     CuMatrix labels = cu.wrapMatrix(msgService.requestMatrix());
//     CuMatrix cuAct =cu.wrapMatrix(activations);
//     CuMatrix cuPredictions = cu.softmaxRows(cuAct);

//     // Check if the label with the highest probability after softmax is equal to the
//     // target label
//     unsigned totalCorrect = cu.checkAccuracy(cuPredictions, labels);

//     // Sum the individual losses of each vertex for this validation partition
//     float lossThisPart = cu.checkLoss(cuPredictions, labels);
// }

void ComputingServer::processBackward(unsigned layer, bool lastLayer) {
    if (lastLayer) {
        gradLoss(layer);
    } else {
        gradLayer(layer);
        if (layer == 0)
            msgService.prefetchWeightsMatrix(totalLayers);
    }
}

void
ComputingServer::gradLayer(unsigned layer) {
    auto tw = gtimers.getTimer("WeightUpdateLayerWait");
    auto tc = gtimers.getTimer("ComputeGradLayer");
    auto tm = gtimers.getTimer("MemcpyBackward2GPU");

    tm->start();
    Matrix grad = gpuComm->oldGradMatrix;
    CuMatrix cuGrad = cu.wrapMatrix(grad);
    Matrix z = gpuComm->savedTensors[layer][TYPE::Z - 1];
    CuMatrix cuZ = cu.wrapMatrix(z);
    tm->stop();

    tc->start();
    CuMatrix interGrad = cu.activateBackward(cuZ, cuGrad);

    Matrix weight = msgService.getWeightMatrix(layer);
    CuMatrix cuWeights = cu.wrapMatrix(weight);
    CuMatrix resultGrad = interGrad.dot(cuWeights, false, true);
    resultGrad.setData(gpuComm->outputTensor.getData());
    resultGrad.updateMatrixFromGPU();
    Matrix ah = gpuComm->savedTensors[layer][TYPE::AH - 1];
    CuMatrix cuAh = cu.wrapMatrix(ah);
    CuMatrix cuWeightUpdates = cuAh.dot(interGrad, true, false);
    Matrix weightUpdates = cuWeightUpdates.getMatrix();
    tc->stop();
    tw->start();
    msgService.sendWeightUpdate(weightUpdates, layer);
    tw->stop();
}


void
ComputingServer::gradLoss(unsigned layer) {
    auto tw = gtimers.getTimer("WeightUpdateLossWait");
    auto tc = gtimers.getTimer("ComputeGradLoss");
    auto tm = gtimers.getTimer("MemcpyBackward2GPU");

    tm->start();
    Matrix predictions = gpuComm->savedTensors[layer][TYPE::ACT - 1];
    Matrix labels = gpuComm->targetTensor;

    CuMatrix cuPredictions = cu.wrapMatrix(predictions);
    CuMatrix cuLabels = cu.wrapMatrix(labels);
    tm->stop();

    tc->start();
    CuMatrix d_output = cu.hadamardSub(cuPredictions, cuLabels);
    Matrix weight = msgService.getWeightMatrix(layer);
    CuMatrix cuWeights = cu.wrapMatrix(weight);
    CuMatrix interGrad = d_output.dot(cuWeights, false, true);
    interGrad.setData(gpuComm->outputTensor.getData());
    interGrad.updateMatrixFromGPU();

    Matrix ah = gpuComm->savedTensors[layer][TYPE::AH - 1];
    CuMatrix cuAh = cu.wrapMatrix(ah);
    CuMatrix cuWeightUpdates = cuAh.dot(d_output, true, false);
    Matrix weightUpdates = cuWeightUpdates.getMatrix();
    tc->stop();

    tw->start();
    msgService.sendWeightUpdate(weightUpdates, layer);
    tw->stop();
}
