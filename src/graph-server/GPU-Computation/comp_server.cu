#include <cmath>
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
// ComputingServer::ComputingServer() : cu(ComputingUnit::getInstance()){};

ComputingServer::ComputingServer(GPUComm *gpu_comm)
    : cu(ComputingUnit::getInstance()),
      msgService(gpu_comm->wPort, gpu_comm->nodeId) {
    gpuComm = gpu_comm;
    totalLayers = gpu_comm->totalLayers;
    nodeId = gpu_comm->nodeId;
    loadWeightServers(weightServerAddrs, gpu_comm->wServersFile);
    msgService.setUpWeightSocket(
        weightServerAddrs.at(nodeId % weightServerAddrs.size()));

    msgService.prefetchWeightsMatrix(totalLayers);
}

// Start listening to main thread
void ComputingServer::terminate() {
    // msgService.terminateWeightServers(weightServerAddrs);
}

// Start GAT-Specific Code
void ComputingServer::vtxNNForward(unsigned layer, bool lastLayer) {
    Matrix feats = (*gpuComm->tensorMap)["h"];
    Matrix weight = msgService.getWeightMatrix(layer);
    auto z = cu.dot(feats, weight);
    memcpy((*gpuComm->tensorMap)["z"].getData(), z.getData(), z.getDataSize());
    delete[] z.getData();
    CuMatrix::freeGPU();
}

void ComputingServer::edgNNForward(unsigned layer, bool lastLayer) {
    cout << "Layer " << layer << endl;
    CuMatrix *adj =
        (CuMatrix *)gpuComm->engine->adjIn;  // any engineer with pursuit should
                                             // not write this;too ugly
    CuMatrix e = *adj;
    int nnz = adj->nnz;
    CuMatrix z = cu.wrapMatrix((*gpuComm->tensorMap)["z"]);
    auto a = cu.wrapMatrix(msgService.getaMatrix(layer));
    auto az = z.dot(a);
    auto act_az = cu.leakyRelu(az, 0.01);
    az.setData((*gpuComm->tensorMap)["az"].getData());
    az.updateMatrixFromGPU();
    CuMatrix e_dst = cu.wrapMatrix(Matrix(1, nnz, (char *)NULL));
    auto cusparseStat = cusparseSgthr(
        cu.spHandle, nnz, act_az.devPtr, e_dst.devPtr, adj->csrRowInd,
        CUSPARSE_INDEX_BASE_ZERO);  // gather the 1st half of az
    assert(CUSPARSE_STATUS_SUCCESS == cusparseStat);
    e_dst.setData((*gpuComm->tensorMap)["A"].getData());
    e_dst.updateMatrixFromGPU();

    CuMatrix::freeGPU();
}

void ComputingServer::edgNNBackward(unsigned layer) {
    CuMatrix a = cu.wrapMatrix(msgService.getaMatrix(layer));

    auto gradTensor = cu.wrapMatrix((*gpuComm->tensorMap)["grad"]);
    auto zaTensor = cu.wrapMatrix((*gpuComm->tensorMap)["az"]);
    unsigned featDim = gradTensor.getCols();
    CuMatrix *adj =
        (CuMatrix *)gpuComm->engine->adjIn;  // any engineer with pursuit should
                                             // not write this;too ugly
    CuMatrix e = *adj;
    unsigned edgCnt = e.nnz;

    cout<<"leakyReluPrime\n";
    CuMatrix d_lrelu = cu.leakyReluPrime(zaTensor, 0.01);  // n x 1
    zaTensor.explicitFree();
    CuMatrix d_lrelu_edge =
        cu.wrapMatrix(Matrix(e.nnz, 1, (char *)NULL));  // BCAST |V| to |E|
    cout<<"cusparseSgthr\n";
    auto cusparseStat = cusparseSgthr(
        cu.spHandle, e.nnz, d_lrelu.devPtr, d_lrelu_edge.devPtr,
        e.csrRowInd,  // Not sure need to see the actually adjmatrix***
        CUSPARSE_INDEX_BASE_ZERO);  // gather the 1st half of az//
    assert(CUSPARSE_STATUS_SUCCESS == cusparseStat);

    vector<int> src_indices(e.nnz);
    vector<int> dst_indices(e.nnz);
    cudaMemcpy(src_indices.data(), e.csrRowInd, sizeof(int) * e.nnz,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(dst_indices.data(), e.csrColInd, sizeof(int) * e.nnz,
               cudaMemcpyDeviceToHost);

    cout<<"gatherRows d_P_edge \n";
    // auto d_P_edge = cu.gatherRows(gradTensor, src_indices);
    auto d_P_edge = cu.gatherRowsGthr(gradTensor, e.csrRowInd,e.nnz);
    cudaDeviceSynchronize();
    cout<<"scaleRowsByVector\n";
    cu.scaleRowsByVector(d_P_edge, d_lrelu_edge);  //(|E|, featDim)
    auto d_Act = d_P_edge;
    d_lrelu.explicitFree();
    cout<<d_Act.shape()<<endl;

    if (layer != 0) {
        // Shape of dA: (|E|, 1), serve as gradient of each edge for backward
        // agg
        cout<<"d_Act.dot(a)\n";
        auto dA = d_Act.dot(a);
        dA.setData((*gpuComm->tensorMap)["dA"].getData());
        dA.updateMatrixFromGPU();
        dA.explicitFree();
    }
    cout<<"d_Act_reduce\n";
    auto d_Act_reduce = cu.reduceColumns(d_Act);
    CuMatrix::MemoryPool.erase(d_Act.devPtr);
    cudaFree(d_Act.devPtr);

    cout<<"gatherRows gatherRows\n";
    auto z = cu.wrapMatrix((*gpuComm->tensorMap)["z"]);
    // auto z_src=cu.gatherRows(z, src_indices);
    // auto z_dst=cu.gatherRows(z, dst_indices);
    auto z_src=cu.gatherRowsGthr(z, e.csrRowInd,e.nnz);
    auto z_dst=cu.gatherRowsGthr(z, e.csrColInd,e.nnz);
    CuMatrix::MemoryPool.erase(z.devPtr);
    cudaDeviceSynchronize();
    cudaFree(z.devPtr);
    cout<<"zz=z_src.dot(z_dst\n";
    auto zz=z_src.dot(z_dst,true,false);
    cout<<"da\n";
    CuMatrix da = zz.dot(d_Act_reduce, false, true);
    da.updateMatrixFromGPU();
    msgService.sendaUpdate(da, layer);
    CuMatrix::freeGPU();
}
void ComputingServer::vtxNNBackward(unsigned layer) {
    auto weight = cu.wrapMatrix(msgService.getWeightMatrix(layer));
    auto grad = cu.wrapMatrix((*gpuComm->tensorMap)["aTg"]);
    auto h = cu.wrapMatrix((*gpuComm->tensorMap)["h"]);
    auto weightUpdates = h.dot(grad, true, false);
    weightUpdates.updateMatrixFromGPU();
    msgService.sendWeightUpdate(weightUpdates, layer);
    weightUpdates.free();

    if (layer != 0) {
        auto resultGrad = grad.dot(weight, false, true);
        resultGrad.setData(
            gpuComm->engine->savedNNTensors[layer - 1]["grad"].getData());
        resultGrad.updateMatrixFromGPU();
        printLog(
            nodeId, "layer %u, resultG %s, output %s", layer,
            resultGrad.shape().c_str(),
            gpuComm->engine->savedNNTensors[layer - 1]["grad"].shape().c_str());
        resultGrad.free();
    }

    CuMatrix::freeGPU();
}

// end GAT

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
    Matrix ah = (*gpuComm->tensorMap)["ah"];
    CuMatrix cuAh = cu.wrapMatrix(ah);

    CuMatrix interGrad = cu.activateBackward(cuAh, cuZ, cuGrad);
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
