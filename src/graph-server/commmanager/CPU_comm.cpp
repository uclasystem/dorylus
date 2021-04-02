#include "CPU_comm.hpp"

#include <omp.h>
using namespace std;
CPUComm::CPUComm(Engine *engine_)
    : engine(engine_), nodeId(engine_->nodeId), totalLayers(engine_->numLayers),
    wServersFile(engine_->weightserverIPFile), wPort(engine_->weightserverPort),
    numNodes(engine_->numNodes),  gnn_type(engine_->gnn_type),
    savedNNTensors(engine_->savedNNTensors),
    msgService(wPort, nodeId, totalLayers, gnn_type)
{
    loadWeightServers(weightServerAddrs, wServersFile);
    msgService.setUpWeightSocket(
        weightServerAddrs.at(nodeId % weightServerAddrs.size()));
    for (char *addr : weightServerAddrs) {
        free(addr);
    }

    msgService.prefetchWeightsMatrix();
}

void CPUComm::NNCompute(Chunk &chunk) {
    unsigned layer = chunk.layer;
    if (chunk.vertex) {
        if (chunk.dir == PROP_TYPE::FORWARD) {
            // printLog(nodeId, "CPU FORWARD vtx NN started");
            vtxNNForward(layer, layer == (totalLayers - 1));
        } else {
            // printLog(nodeId, "CPU BACKWARD vtx NN started");
            vtxNNBackward(layer);
        }
    } else {
        layer--; // YIFAN: fix this
        if (chunk.dir == PROP_TYPE::FORWARD) {
            // printLog(nodeId, "CPU FORWARD edg NN started");
            edgNNForward(layer, layer == (totalLayers - 1));
        } else {
            // printLog(nodeId, "CPU BACKWARD edg NN started");
            edgNNBackward(layer);
        }
    }
    // printLog(nodeId, "CPU NN Done");
    NNRecvCallback(engine, chunk);
}

void CPUComm::vtxNNForward(unsigned layer, bool lastLayer) {
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

void CPUComm::vtxNNBackward(unsigned layer) {
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

void CPUComm::edgNNForward(unsigned layer, bool lastLayer) {
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

void CPUComm::edgNNBackward(unsigned layer) {
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

void CPUComm::vtxNNForwardGCN(unsigned layer, bool lastLayer) {
    Matrix feats = savedNNTensors[layer]["ah"];
    Matrix weight = msgService.getWeightMatrix(layer);
    Matrix z = feats.dot(weight);
    if (!lastLayer) {
        memcpy(savedNNTensors[layer]["z"].getData(), z.getData(), z.getDataSize());
        Matrix act_z = activate(z);  // z data get activated ...
        memcpy(savedNNTensors[layer]["h"].getData(), act_z.getData(),
               act_z.getDataSize());
        deleteMatrix(act_z);
    } else {
        Matrix predictions = softmax(z);
        Matrix labels = savedNNTensors[layer]["lab"];

        float acc, loss;
        getTrainStat(predictions, labels, acc, loss);
        unsigned valsetSize = (unsigned)(predictions.getRows() * VAL_PORTION);
        msgService.sendAccloss(acc, loss, predictions.getRows());
        printLog(nodeId, "batch Acc: %f, Loss: %f", acc / valsetSize, loss / valsetSize);

        maskout(predictions, labels);

        Matrix d_output = hadamardSub(predictions, labels);
        d_output /= engine->graph.globalVtxCnt * TRAIN_PORTION; // Averaging init backward gradient
        Matrix weight = msgService.getWeightMatrix(layer);
        Matrix interGrad = d_output.dot(weight, false, true);
        memcpy(savedNNTensors[layer]["grad"].getData(), interGrad.getData(),
               interGrad.getDataSize());

        Matrix ah = savedNNTensors[layer]["ah"];
        Matrix weightUpdates = ah.dot(d_output, true, false);
        msgService.sendWeightUpdate(weightUpdates, layer);
        deleteMatrix(interGrad);
        deleteMatrix(d_output);
        deleteMatrix(predictions);
    }
    deleteMatrix(z);
}

void CPUComm::vtxNNBackwardGCN(unsigned layer) {
    Matrix weight = msgService.getWeightMatrix(layer);
    Matrix grad = savedNNTensors[layer]["aTg"];
    Matrix z = savedNNTensors[layer]["z"];

    Matrix actDeriv = activateDerivative(z);
    Matrix interGrad = grad * actDeriv;

    Matrix ah = savedNNTensors[layer]["ah"];
    Matrix weightUpdates = ah.dot(interGrad, true, false);
    msgService.sendWeightUpdate(weightUpdates, layer);
    if (layer != 0) {
        Matrix resultGrad = interGrad.dot(weight, false, true);
        memcpy(savedNNTensors[layer]["grad"].getData(), resultGrad.getData(),
               resultGrad.getDataSize());
        deleteMatrix(resultGrad);
    }

    deleteMatrix(actDeriv);
    deleteMatrix(interGrad);

    if (layer == 0) msgService.prefetchWeightsMatrix();
}

void CPUComm::vtxNNForwardGAT(unsigned layer, bool lastLayer) {
    Matrix feats = layer == 0
                 ? savedNNTensors[layer]["h"]
                 : savedNNTensors[layer - 1]["ah"];
    Matrix weight = msgService.getWeightMatrix(layer);
    Matrix z = feats.dot(weight);
    memcpy(savedNNTensors[layer]["z"].getData(), z.getData(), z.getDataSize());
    deleteMatrix(z);
}

void CPUComm::vtxNNBackwardGAT(unsigned layer) {
    Matrix weight = msgService.getWeightMatrix(layer);
    Matrix grad = savedNNTensors[layer]["aTg"];
    Matrix h = layer == 0
             ? savedNNTensors[layer]["h"]
             : savedNNTensors[layer - 1]["ah"];

    Matrix weightUpdates = h.dot(grad, true, false);
    msgService.sendWeightUpdate(weightUpdates, layer);

    if (layer != 0) {
        Matrix resultGrad = grad.dot(weight, false, true);
        memcpy(savedNNTensors[layer - 1]["grad"].getData(), resultGrad.getData(),
               resultGrad.getDataSize());
        resultGrad.free();
    }
    if (layer == 0) msgService.prefetchWeightsMatrix();
}

void CPUComm::edgNNForwardGAT(unsigned layer, bool lastLayer) {
    Matrix a = msgService.getaMatrix(layer); // YIFAN: fix this
    unsigned featLayer = layer; // YIFAN: fix this
    Matrix z = savedNNTensors[featLayer]["z"];

    // expand and dot
    Matrix zaTensor = expandDot(z, a, engine->graph.forwardAdj);
    memcpy(savedNNTensors[featLayer]["az"].getData(), zaTensor.getData(), zaTensor.getDataSize());
    Matrix outputTensor = leakyRelu(zaTensor);
    zaTensor.free();

    memcpy(savedNNTensors[featLayer]["A"].getData(), outputTensor.getData(), outputTensor.getDataSize());
    outputTensor.free();
}

void CPUComm::edgNNBackwardGAT(unsigned layer) {
    Matrix a = msgService.getaMatrix(layer);
    unsigned featLayer = layer;
    Matrix gradTensor = savedNNTensors[featLayer]["grad"];
    Matrix zaTensor = savedNNTensors[featLayer]["az"];
    Matrix localZTensor = savedNNTensors[featLayer]["z"]; // serve as Z_dst, and part of Z_src
    // Matrix ghostZTensor = savedNNTensors[featLayer]["fg_z"]; // serve as part of Z_src
    // FeatType **fedge = engine->savedEdgeTensors[featLayer]["fedge"]; // This serves the purpose of Z_src and Z_dst
    // unsigned edgCnt = engine->graph.forwardAdj.nnz;
    // unsigned featDim = gradTensor.getCols();

    // gradient of LeakyRelu
    Matrix dLRelu = leakyReluBackward(zaTensor);
    // expand dP to (|E|, featDim) and element-wise multiply dLRelu
    // Shape of dAct is (|E|, featDim)
    Matrix dAct = expandHadamardMul(gradTensor, dLRelu, engine->graph.forwardAdj);
    dLRelu.free();

    // Shape of dA: (|E|, 1), serve as gradient of each edge for backward agg
    Matrix dA = dAct.dot(a);
    memcpy(savedNNTensors[featLayer]["dA"].getData(), dA.getData(), dA.getDataSize());
    dA.free();

    // reduce dAct(|E|, featDim) to (1, featDim)
    // Do this first to optmize computation and save memory
    Matrix dAct_reduce = reduce(dAct);
    dAct.free();
    // // Expand Z_src and Z_dst (both have shape (|V|, featDim)) to (|E|, featDim)
    // // And then do Z_dst^T \dot Z_src -> zz (featDim, featDim)
    // Matrix zz = expandMulZZ(fedge, edgCnt, featDim);
    Matrix zz = localZTensor.dot(localZTensor, true, false);
    // (1, featDim) \dot (featDim, featDim) -> (1, featDim), which is da's shape
    Matrix da = zz.dot(dAct_reduce, false, true);
    dAct_reduce.free();
    zz.free();
    msgService.sendaUpdate(da, layer);
    // da.free();
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

Matrix activate(Matrix &mat) {
    FeatType *activationData = new FeatType[mat.getNumElemts()];
    FeatType *zData = mat.getData();

#pragma omp parallel for
    for (unsigned i = 0; i < mat.getNumElemts(); ++i)
        activationData[i] = std::tanh(zData[i]);

    return Matrix(mat.getRows(), mat.getCols(), activationData);
}

Matrix softmax(Matrix &mat) {
    FeatType *result = new FeatType[mat.getNumElemts()];

#pragma omp parallel for
    for (unsigned r = 0; r < mat.getRows(); ++r) {
        unsigned length = mat.getCols();
        FeatType *vecSrc = mat.getData() + r * length;
        FeatType *vecDst = result + r * length;

        FeatType denom = 1e-20;
        FeatType maxEle = *(std::max_element(vecSrc, vecSrc + length));
        for (unsigned c = 0; c < length; ++c) {
            vecDst[c] = std::exp(vecSrc[c] - maxEle);
            denom += vecDst[c];
        }
        for (unsigned c = 0; c < length; ++c) {
            vecDst[c] /= denom;
        }
    }

    return Matrix(mat.getRows(), mat.getCols(), result);
}

Matrix expandDot(Matrix &m, Matrix &v, CSCMatrix<EdgeType> &forwardAdj) {
    FeatType *outputData = new FeatType[forwardAdj.nnz];
    Matrix outputTensor(forwardAdj.nnz, 1, outputData);
    memset(outputData, 0, outputTensor.getDataSize());

    unsigned vtcsCnt = m.getRows();
    unsigned featDim = m.getCols();
    FeatType *vPtr = v.getData();
#pragma omp parallel for
    for (unsigned lvid = 0; lvid < vtcsCnt; lvid++) {
        FeatType *mPtr = m.get(lvid);
        for (unsigned long long eid = forwardAdj.columnPtrs[lvid];
            eid < forwardAdj.columnPtrs[lvid + 1]; ++eid) {
            for (unsigned j = 0; j < featDim; ++j) {
                outputData[eid] += mPtr[j] * vPtr[j];
            }
        }
    }

    return outputTensor;
}

Matrix expandHadamardMul(Matrix &m, Matrix &v, CSCMatrix<EdgeType> &forwardAdj) {
    unsigned vtcsCnt = m.getRows();
    unsigned featDim = m.getCols();
    unsigned edgCnt = forwardAdj.nnz;

    FeatType *outputData = new FeatType[edgCnt * featDim];
    Matrix outputTensor(forwardAdj.nnz, featDim, outputData);
    memset(outputData, 0, outputTensor.getDataSize());

    FeatType *vPtr = v.getData();
#pragma omp parallel for
    for (unsigned lvid = 0; lvid < vtcsCnt; lvid++) {
        FeatType *mPtr = m.get(lvid);
        for (unsigned long long eid = forwardAdj.columnPtrs[lvid];
            eid < forwardAdj.columnPtrs[lvid + 1]; ++eid) {
            FeatType normFactor = vPtr[eid];
            for (unsigned j = 0; j < featDim; ++j) {
                outputData[eid * featDim + j] = mPtr[j] * normFactor;
            }
        }
    }

    return outputTensor;
}

Matrix expandMulZZ(FeatType **eFeats, unsigned edgCnt, unsigned featDim) {
    FeatType *zzData = new FeatType[featDim * featDim];
    Matrix zzTensor(featDim, featDim, zzData);
    memset(zzData, 0, zzTensor.getDataSize());

    FeatType **srcFeats = eFeats;
    FeatType **dstFeats = eFeats + edgCnt;

#pragma omp parallel for
    for (unsigned i = 0; i < featDim; i++) {
        for (unsigned eid = 0; eid < edgCnt; eid++) {
            for (unsigned j = 0; j < featDim; ++j) {
                zzData[i * featDim + j] += srcFeats[eid][i] * dstFeats[eid][j];
            }
        }
    }

    return zzTensor;
}

Matrix reduce(Matrix &mat) {
    unsigned edgCnt = mat.getRows();
    unsigned featDim = mat.getCols();

    FeatType *outputData = new FeatType[featDim];
    Matrix outputTensor(1, featDim, outputData);

    FeatType *mPtr = mat.getData();
#pragma omp parallel for
    for (unsigned i = 0; i < featDim; i++) {
        for (unsigned eid = 0; eid < edgCnt; eid++) {
            outputData[i] += mPtr[eid * featDim + i];
        }
    }

    return outputTensor;
}

Matrix leakyRelu(Matrix &mat) {
    FeatType alpha = 0.01;
    FeatType *activationData = new FeatType[mat.getNumElemts()];
    FeatType *inputData = mat.getData();

#pragma omp parallel for
    for (unsigned i = 0; i < mat.getNumElemts(); ++i) {
        activationData[i] = (inputData[i] > 0) ? inputData[i] : alpha * inputData[i];
    }

    return Matrix(mat.getRows(), mat.getCols(), activationData);
}

Matrix leakyReluBackward(Matrix &mat) {
    FeatType alpha = 0.01;
    FeatType *outputData = new FeatType[mat.getNumElemts()];
    FeatType *inputData = mat.getData();

#pragma omp parallel for
    for (unsigned i = 0; i < mat.getNumElemts(); ++i) {
        outputData[i] = (inputData[i] > 0) ? 1 : alpha;
    }

    return Matrix(mat.getRows(), mat.getCols(), outputData);
}

Matrix hadamardMul(Matrix &A, Matrix &B) {
    FeatType *result = new FeatType[A.getRows() * A.getCols()];

    FeatType *AData = A.getData();
    FeatType *BData = B.getData();

#pragma omp parallel for
    for (unsigned ui = 0; ui < A.getNumElemts(); ++ui) {
        result[ui] = AData[ui] * BData[ui];
    }

    return Matrix(A.getRows(), A.getCols(), result);
}

Matrix hadamardSub(Matrix &A, Matrix &B) {
    FeatType *result = new FeatType[A.getRows() * B.getCols()];

    FeatType *AData = A.getData();
    FeatType *BData = B.getData();

#pragma omp parallel for
    for (unsigned ui = 0; ui < A.getNumElemts(); ++ui)
        result[ui] = AData[ui] - BData[ui];

    return Matrix(A.getRows(), B.getCols(), result);
}

Matrix activateDerivative(Matrix &mat) {
    FeatType *res = new FeatType[mat.getNumElemts()];
    FeatType *zData = mat.getData();

#pragma omp parallel for
    for (unsigned i = 0; i < mat.getNumElemts(); ++i)
        res[i] = 1 - std::pow(std::tanh(zData[i]), 2);

    return Matrix(mat.getRows(), mat.getCols(), res);
}

void CPUComm::getTrainStat(Matrix &preds, Matrix &labels, float &acc,
                           float &loss) {
    acc = 0.0;
    loss = 0.0;
    unsigned featDim = labels.getCols();
    unsigned valStt = (unsigned)(labels.getRows() * TRAIN_PORTION);
    unsigned valEnd = valStt + (unsigned)(labels.getRows() * VAL_PORTION);
    for (unsigned i = valStt; i < valEnd; i++) {
        FeatType *currLabel = labels.getData() + i * labels.getCols();
        FeatType *currPred = preds.getData() + i * labels.getCols();
        acc += currLabel[argmax(currPred, currPred + featDim)];
        loss -= std::log(currPred[argmax(currLabel, currLabel + featDim)]);
    }
    // printLog(nodeId, "batch loss %f, batch acc %f", loss, acc);
}

void CPUComm::maskout(Matrix &preds, Matrix &labels) {
    unsigned end = labels.getRows();
    unsigned stt = (unsigned)(end * TRAIN_PORTION);

    FeatType *predStt = preds.get(stt);
    FeatType *labelStt = labels.get(stt);
    memcpy(predStt, labelStt, sizeof(FeatType) * (end - stt));
}

void deleteMatrix(Matrix &mat) {
    if (!mat.empty()) {
        delete[] mat.getData();
        mat = Matrix();
    }
}