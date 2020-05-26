#include "CPU_comm.hpp"

#include <omp.h>
using namespace std;

CPUComm::CPUComm(Engine *engine_)
    : engine(engine_), nodeId(engine_->nodeId), totalLayers(engine_->numLayers),
    wServersFile(engine_->weightserverIPFile), wPort(engine_->weightserverPort),
    numNodes(engine_->numNodes), currLayer(0), savedNNTensors(engine_->savedNNTensors),
    msgService(wPort, nodeId)
{
    loadWeightServers(weightServerAddrs, wServersFile);
    msgService.setUpWeightSocket(
        weightServerAddrs.at(nodeId % weightServerAddrs.size()));
}

void CPUComm::NNCompute(Chunk &chunk) {
    c=chunk;
    currLayer = chunk.layer;

    if (chunk.dir == PROP_TYPE::FORWARD) {
        if (chunk.vertex) {
            printLog(nodeId, "CPU FORWARD vtx NN started");
            vtxNNForward(currLayer, currLayer == (totalLayers - 1));
        } else {
            printLog(nodeId, "CPU FORWARD edg NN started");
            edgNNForward(currLayer, currLayer == (totalLayers - 1));
        }
    }
    if (chunk.dir == PROP_TYPE::BACKWARD) {
        if (chunk.vertex) {
            printLog(nodeId, "CPU BACKWARD vtx NN started");
            vtxNNBackward(currLayer);
        } else {
            printLog(nodeId, "CPU BACKWARD edg NN started");
            edgNNBackward(currLayer);
        }
    }
    printLog(nodeId, "CPU NN Done");
}

void CPUComm::vtxNNForward(unsigned layer, bool lastLayer) {
    Matrix feats = savedNNTensors[layer]["h"];
    Matrix weight = msgService.getWeightMatrix(layer);

    Matrix z = feats.dot(weight);

    printLog(nodeId, "layer %u, feats %s, weight %s, z %s, output %s", layer,
        feats.shape().c_str(), weight.shape().c_str(), z.shape().c_str(), savedNNTensors[layer]["z"].shape().c_str());

    memcpy(savedNNTensors[layer]["z"].getData(), z.getData(), z.getDataSize());
    deleteMatrix(z);
}

void CPUComm::vtxNNBackward(unsigned layer) {
    Matrix weight = msgService.getWeightMatrix(layer);
    Matrix grad = savedNNTensors[layer]["aTg"];
    Matrix h = savedNNTensors[layer]["h"];

    printLog(nodeId, "layer %u, weight %s, grad %s, h %s", layer,
        weight.shape().c_str(), grad.shape().c_str(), h.shape().c_str());

    Matrix weightUpdates = h.dot(grad, true, false);
    msgService.sendWeightUpdate(weightUpdates, layer);
    printLog(nodeId, "layer %u, weight %s, grad %s, h %s, wu %s", layer,
        weight.shape().c_str(), grad.shape().c_str(), h.shape().c_str(), weightUpdates.shape().c_str());

    deleteMatrix(weightUpdates);

    if (layer != 0) {
        Matrix resultGrad = grad.dot(weight, false, true);
        memcpy(savedNNTensors[layer - 1]["grad"].getData(), resultGrad.getData(),
               resultGrad.getDataSize());
        printLog(nodeId, "layer %u, resultG %s, output %s", layer, resultGrad.shape().c_str(), savedNNTensors[layer - 1]["grad"].shape().c_str());
        deleteMatrix(resultGrad);
    }
}

void CPUComm::edgNNForward(unsigned layer, bool lastLayer) {
    // Matrix feats = (*tensorMap)["h"];
    // Matrix weight = msgService.getWeightMatrix(layer);

    // Matrix z = feats.dot(weight);
    // memcpy((*tensorMap)["z"].getData(), z.getData(), z.getDataSize());
    // deleteMatrix(z);
}

void CPUComm::edgNNBackward(unsigned layer) {
    // Matrix weight = msgService.getWeightMatrix(layer);
    // Matrix grad = (*tensorMap)["aTg"];
    // Matrix h = (*tensorMap)["h"];

    // Matrix weightUpdates = h.dot(grad, true, false);
    // msgService.sendWeightUpdate(weightUpdates, layer);
    // deleteMatrix(weightUpdates);

    // if (layer != 0) {
    //     Matrix resultGrad = grad.dot(weight, false, true);
    //     memcpy((*tensorMap)["grad"].getData(), resultGrad.getData(),
    //            resultGrad.getDataSize());
    //     deleteMatrix(resultGrad);
    // }
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
    for (unsigned i = 0; i < labels.getRows(); i++) {
        FeatType *currLabel = labels.getData() + i * labels.getCols();
        FeatType *currPred = preds.getData() + i * labels.getCols();
        acc += currLabel[argmax(currPred, currPred + featDim)];
        loss -= std::log(currPred[argmax(currLabel, currLabel + featDim)]);
    }
    acc /= labels.getRows();
    loss /= labels.getRows();
    printLog(nodeId, "batch loss %f, batch acc %f", loss, acc);
}

void deleteMatrix(Matrix &mat) {
    if (!mat.empty()) {
        delete[] mat.getData();
        mat = Matrix();
    }
}