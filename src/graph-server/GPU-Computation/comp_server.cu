#include <cmath>
#include <iostream>
#include <thread>

#include "../../common/utils.hpp"
#include "../utils/utils.hpp"
#include "comp_server.cuh"

void deleteMatrix(Matrix &mat) {
    if (!mat.empty()) {
        delete[] mat.getData();
        mat = Matrix();
    }
}

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

ComputingServer::ComputingServer(GPUComm *gpu_comm, GNN gnn_type_, int devId, ComputingUnit& cu_)
    : cu(cu_), gpuComm(gpu_comm),
    totalLayers(gpu_comm->totalLayers), nodeId(gpu_comm->nodeId),
    gnn_type(gnn_type_), savedNNTensors(gpu_comm->savedNNTensors),
    msgService(gpu_comm->msgService), deviceId(devId)
{
    loadWeightServers(weightServerAddrs, gpu_comm->wServersFile);
    msgService.setUpWeightSocket(
        weightServerAddrs.at(nodeId % weightServerAddrs.size()));
    for (char *addr : weightServerAddrs) {
        free(addr);
    }

    msgService.prefetchWeightsMatrix();
}

void ComputingServer::vtxNNForward(unsigned layer, bool lastLayer) {
    switch (gnn_type) {
        case GNN::GCN:
            vtxNNForwardGCN(layer, lastLayer);
            break;
        case GNN::GAT:
            vtxNNForwardGAT(layer, lastLayer);
            break;
        default:
            abort();
    }
}

void ComputingServer::vtxNNBackward(unsigned layer) {
    switch (gnn_type) {
        case GNN::GCN:
            vtxNNBackwardGCN(layer);
            break;
        case GNN::GAT:
            vtxNNBackwardGAT(layer);
            break;
        default:
            abort();
    }
}

void ComputingServer::edgNNForward(unsigned layer, bool lastLayer) {
    switch (gnn_type) {
        case GNN::GCN:
            edgNNForwardGCN(layer, lastLayer);
            break;
        case GNN::GAT:
            edgNNForwardGAT(layer, lastLayer);
            break;
        default:
            abort();
    }
}

void ComputingServer::edgNNBackward(unsigned layer) {
    switch (gnn_type) {
        case GNN::GCN:
            edgNNBackwardGCN(layer);
            break;
        case GNN::GAT:
            edgNNBackwardGAT(layer);
            break;
        default:
            abort();
    }
}

void ComputingServer::vtxNNForwardGCN(unsigned layer, bool lastLayer) {
    double avStart = getTimer();
    CuMatrix cuFeats = ahs[layer];
    Matrix weights = msgService.getWeightMatrix(layer);
    CuMatrix cuWeights = cu.wrapMatrix(weights);
    double avComputeZ = getTimer();
    CuMatrix z = cuFeats.dot(cuWeights);

    if (!lastLayer) {
        double avLastLayer = getTimer();
        Matrix outputTensor = savedNNTensors[layer]["h"];
        FeatType *act_z = outputTensor.getData();
        // z.setData(z_data);
//        z.updateMatrixFromGPU();
//        memcpy(z_data, z.getData(), z.getDataSize());
        CuMatrix h = cu.activate(z);  // z data get activated ...
        // z.setData(act_z);
        h.setData(act_z);
        h.updateMatrixFromGPU();
        //memcpy(act_z, z.getData(), z.getDataSize());

        zs[layer] = z;
        hs[layer] = h;

        std::map<std::string, double>& timesMap = gpuComm->engine->applyTimes[layer];
        addOrCreate("1. Wrap Feats/Weights", avComputeZ - avStart, timesMap);
        addOrCreate("2. Compute Z", avLastLayer - avComputeZ, timesMap);
        addOrCreate("3. Last Layer Ops", getTimer() - avLastLayer, timesMap);
    } else {
        bool report = true;
        double avPredict = getTimer();

        CuMatrix cuPred = cu.softmaxRows(z);
        // here it can be optimized by fetching directly from Forward;
        Matrix labels = savedNNTensors[layer]["lab"];
        CuMatrix cuLabels = cu.wrapMatrix(labels);
        if (report) {
            // Asynchronously do acc, loss calc on CPUs
            std::thread evalThread([&](Matrix labels, unsigned layer) {
                double reportStart = getTimer();
                cudaError_t err = cudaSetDevice(deviceId);
                if (err != cudaSuccess) {
                    abort();
                }
                Matrix cpuPreds = cuPred.getMatrix();
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
                // printLog(nodeId, "ACC %f, LOSS %f", acc, loss);
                msgService.sendAccloss(acc, loss, cpuPreds.getRows());
                unsigned valsetSize = (unsigned)(cpuPreds.getRows() * VAL_PORTION);
                printLog(nodeId, "batch Acc: %f, Loss: %f", acc / valsetSize, loss / valsetSize);
                cpuPreds.free();

                double reportEnd = getTimer();
                std::map<std::string, double>& reportTimesMap = gpuComm->engine->applyTimes[layer];
                addOrCreate("Report", reportEnd - reportStart, reportTimesMap);
            },
            labels, layer);
            // evalThread.join();
            evalThread.detach();
            // float acc, loss;
            // cu.getTrainStat(cuPred, cuLabels, acc, loss);
            // unsigned valsetSize = (unsigned)(cuPred.getRows() * VAL_PORTION);
            // msgService.sendAccloss(acc, loss, cuPred.getRows());
            // // printLog(nodeId, "valset size %u, total size %u", valsetSize, cuPred.getRows());
            // printLog(nodeId, "batch Acc: %f, Loss: %f", acc / valsetSize, loss / valsetSize);
        }
        double calcUpdates = getTimer();
        cu.maskout(cuPred, cuLabels);

        CuMatrix d_output = cu.hadamardSub(cuPred, cuLabels);
        d_output.scale(1.0 / (gpuComm->engine->graph.globalVtxCnt * TRAIN_PORTION));

        CuMatrix interGrad = d_output.dot(cuWeights, false, true);
        grads[layer] = interGrad;
        interGrad.setData(savedNNTensors[layer]["grad"].getData());
        interGrad.updateMatrixFromGPU();

        //Matrix ah = savedNNTensors[layer]["ah"];
        CuMatrix cuAh = ahs[layer]; //cu.wrapMatrix(ah);
        CuMatrix cuWeightUpdates = cuAh.dot(d_output, true, false);
        Matrix weightUpdates = cuWeightUpdates.getMatrix();
        double sendUpdates = getTimer();
        msgService.sendWeightUpdate(weightUpdates, layer);

        std::map<std::string, double>& timesMap = gpuComm->engine->applyTimes[layer];
        addOrCreate("1. WrapFeats/Weights", avComputeZ - avStart, timesMap);
        addOrCreate("2. Compute Z", avPredict - avComputeZ, timesMap);
        addOrCreate("3. Make Predictions", calcUpdates - avPredict, timesMap);
        addOrCreate("4. Calc Updates", sendUpdates - calcUpdates, timesMap);
        addOrCreate("5. Send Updates", getTimer() - sendUpdates, timesMap);
    }

}

void ComputingServer::vtxNNBackwardGCN(unsigned layer) {
    cudaError_t err = cudaSetDevice(deviceId);
    if (err != cudaSuccess) {
        abort();
    }

    double avbStart = getTimer();
    //Matrix grad = savedNNTensors[layer]["aTg"];
    CuMatrix cuGrad = aTgs[layer];
    //Matrix z = savedNNTensors[layer]["z"];
    CuMatrix cuZ = zs[layer]; //cu.wrapMatrix(z);
    //Matrix h = savedNNTensors[layer]["h"];
    CuMatrix cuH = hs[layer]; //cu.wrapMatrix(h);
    //Matrix ah = savedNNTensors[layer]["ah"];
    CuMatrix cuAh = ahs[layer]; //cu.wrapMatrix(ah);

    double avbCompute = getTimer();
    CuMatrix interGrad = cu.activateBackward(cuZ, cuH, cuGrad);
    CuMatrix cuWeightUpdates = cuAh.dot(interGrad, true, false);

    double avbGetUpdates = getTimer();
    Matrix weightUpdates = cuWeightUpdates.getMatrix();
    Matrix weight = msgService.getWeightMatrix(layer);
    CuMatrix cuWeights = cu.wrapMatrix(weight);
    double avbComputeGrad = getTimer();
    if (layer != 0) {
        CuMatrix resultGrad = interGrad.dot(cuWeights, false, true);
        resultGrad.setData(savedNNTensors[layer]["grad"].getData());
        resultGrad.updateMatrixFromGPU();
    }

    double avbSendUpdates = getTimer();
    msgService.sendWeightUpdate(weightUpdates, layer);
    //CuMatrix::freeGPU();

    if (layer == 0) msgService.prefetchWeightsMatrix();

    unsigned numLayers = gpuComm->engine->numLayers;
    std::map<std::string, double>& timesMap = gpuComm->engine->applyTimes[layer + numLayers];
    addOrCreate("1. Wrap to CUDA", avbCompute - avbStart, timesMap);
    addOrCreate("2. Compute Updates", avbGetUpdates - avbCompute, timesMap);
    addOrCreate("3. Get Weights", avbComputeGrad - avbGetUpdates, timesMap);
    addOrCreate("4. Compute Grad", avbSendUpdates - avbComputeGrad, timesMap);
    addOrCreate("5. Send Weight Updates", getTimer() - avbSendUpdates, timesMap);
}

void ComputingServer::vtxNNForwardGAT(unsigned layer, bool lastLayer) {
    Matrix feats = layer == 0
                 ? savedNNTensors[layer]["h"]
                 : savedNNTensors[layer - 1]["ah"];
    Matrix weight = msgService.getWeightMatrix(layer);
    CuMatrix cu_h = cu.wrapMatrix(feats);
    CuMatrix cu_w = cu.wrapMatrix(weight);
    CuMatrix z = cu_h.dot(cu_w);
    z.setData(savedNNTensors[layer]["z"].getData());
    z.updateMatrixFromGPU();

    CuMatrix::freeGPU();
}

void ComputingServer::vtxNNBackwardGAT(unsigned layer) {
    Matrix host_h = layer == 0
                  ? savedNNTensors[layer]["h"]
                  : savedNNTensors[layer - 1]["ah"];

    auto weight = cu.wrapMatrix(msgService.getWeightMatrix(layer));
    auto grad = cu.wrapMatrix(savedNNTensors[layer]["aTg"]);
    auto h = cu.wrapMatrix(host_h);
    auto weightUpdates = h.dot(grad, true, false);
    // std::cout << "weightUpdates " << weightUpdates.shape() << std::endl;
    // weightUpdates.updateMatrixFromGPU();
    Matrix host_wu = weightUpdates.getMatrix();
    msgService.sendWeightUpdate(host_wu, layer);

    if (layer != 0) {
        auto resultGrad = grad.dot(weight, false, true);
        resultGrad.setData(
            gpuComm->engine->savedNNTensors[layer - 1]["grad"].getData());
        resultGrad.updateMatrixFromGPU();
        // printLog(
        //     nodeId, "layer %u, resultG %s, output %s", layer,
        //     resultGrad.shape().c_str(),
        //     gpuComm->engine->savedNNTensors[layer - 1]["grad"].shape().c_str());
    }

    CuMatrix::freeGPU();
    if (layer == 0) msgService.prefetchWeightsMatrix();
}

void ComputingServer::edgNNForwardGAT(unsigned layer, bool lastLayer) {
    auto a = cu.wrapMatrix(msgService.getaMatrix(layer));
    unsigned featLayer = layer; // YIFAN: fix this
    CuMatrix z = cu.wrapMatrix(savedNNTensors[featLayer]["z"]);
    // YIFAN: check this
    CuMatrix e = *NormAdjMatrixIn;
    // int nnz = e.nnz;
    auto az = z.dot(a);
    az.setData(savedNNTensors[featLayer]["az"].getData());
    az.updateMatrixFromGPU();
    CuMatrix e_dst = cu.wrapMatrix(Matrix(1, e.nnz, (char *)NULL));
    auto cusparseStat = cusparseSgthr(cu.spHandle, e.nnz, az.devPtr, e_dst.devPtr,
                                      e.csrRowInd, CUSPARSE_INDEX_BASE_ZERO);
    assert(CUSPARSE_STATUS_SUCCESS == cusparseStat);

    auto act_edge = cu.leakyRelu(e_dst, 0.01);
    e_dst.explicitFree();
    act_edge.setData(savedNNTensors[featLayer]["A"].getData());
    act_edge.updateMatrixFromGPU();
    CuMatrix::freeGPU();
}

void ComputingServer::edgNNBackwardGAT(unsigned layer) {
    unsigned featLayer = layer;
    auto zaTensor = cu.wrapMatrix(savedNNTensors[featLayer]["az"]);
    // YIFAN: and check this
    CuMatrix e = *NormAdjMatrixIn;
    // unsigned edgCnt = e.nnz;

    CuMatrix az_edge = cu.wrapMatrix(Matrix(e.nnz, 1, (char *)NULL));
    auto cusparseStat = cusparseSgthr(
        cu.spHandle, e.nnz, zaTensor.devPtr, az_edge.devPtr,
        e.csrRowInd,  // Not sure, need to see the actually adjmatrix***
        CUSPARSE_INDEX_BASE_ZERO);  // gather the 1st half of az//
    assert(CUSPARSE_STATUS_SUCCESS == cusparseStat);

    CuMatrix d_lrelu_edge = cu.leakyReluPrime(az_edge, 0.01);  // nnz x 1
    az_edge.explicitFree();

    // std::cout << "gatherRows d_P_edge \n";
    auto gradTensor = cu.wrapMatrix(savedNNTensors[featLayer]["grad"]);
    // std::cout << "gradTensor.shape " << gradTensor.shape() << std::endl;
    // std::cout << "e.nnz " << e.nnz << std::endl;
    auto d_P_edge = cu.gatherRowsGthr(gradTensor, e.csrRowInd, e.nnz);
    gradTensor.explicitFree();

    // std::cout << "scaleRowsByVector\n";
    cu.scaleRowsByVector(d_P_edge, d_lrelu_edge);  //(|E|, featDim)
    auto d_Act = d_P_edge;
    d_lrelu_edge.explicitFree();
    // std::cout << "d_Act.shape() " << d_Act.shape() << std::endl

    if (layer != 0) {
        CuMatrix a = cu.wrapMatrix(msgService.getaMatrix(layer));
        // Shape of dA: (|E|, 1), serve as gradient of each edge for backward
        // agg
        auto dA = d_Act.dot(a);
        dA.setData(savedNNTensors[featLayer]["dA"].getData());
        dA.updateMatrixFromGPU();
        dA.explicitFree();
    }
    // std::cout << "d_Act_reduce\n";
    auto d_Act_reduce = cu.reduceColumns(d_Act);
    d_Act.explicitFree();

    // std::cout << "gatherRows\n";
    auto z = cu.wrapMatrix(savedNNTensors[featLayer]["z"]);
    // std::cout << "zz=z.dot(z)\n";
    auto zz = z.dot(z, true, false);
    // std::cout << "da\n";
    CuMatrix da = zz.dot(d_Act_reduce, false, true);
    // da.updateMatrixFromGPU();
    Matrix host_da = da.getMatrix();
    msgService.sendaUpdate(host_da, layer);
    CuMatrix::freeGPU();
}
