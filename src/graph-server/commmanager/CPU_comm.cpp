#include "CPU_comm.hpp"

#include <omp.h>
using namespace std;
CPUComm::CPUComm(Engine *engine_) : ResourceComm() {
    engine = engine_;
    nodeId = engine->nodeId;
    totalLayers = engine->numLayers;
    wServersFile = engine->weightserverIPFile;
    wPort = engine->weightserverPort;
    numNodes = engine->numNodes;
    currLayer = 0;

    msgService = MessageService(wPort, nodeId);
    loadWeightServers(weightServerAddrs, wServersFile);
    msgService.setUpWeightSocket(
        weightServerAddrs.at(nodeId % weightServerAddrs.size()));

    msgService.prefetchWeightsMatrix(totalLayers);
}
void CPUComm::NNCompute(Chunk &chunk) {
    c=chunk;
    currLayer = chunk.layer;
    tensorMap = &engine->savedNNTensors[chunk.layer];

    if (chunk.dir == PROP_TYPE::FORWARD) {
        // printLog(nodeId, "CPU FORWARD NN started");
        processForward(currLayer, currLayer == (totalLayers - 1));
    }
    if (chunk.dir == PROP_TYPE::BACKWARD) {
        // printLog(nodeId, "CPU BACKWARD NN started");
        processBackward(currLayer);
    }
    // printLog(nodeId, "CPU NN Done");
}

void CPUComm::processForward(unsigned layer, bool lastLayer) {
    Matrix feats = (*tensorMap)["ah"];
    Matrix weight = msgService.getWeightMatrix(currLayer);
    Matrix z = feats.dot(weight);
    if (!lastLayer) {
        memcpy((*tensorMap)["z"].getData(), z.getData(), z.getDataSize());
        Matrix act_z = activate(z);  // z data get activated ...
        memcpy((*tensorMap)["h"].getData(), act_z.getData(),
               act_z.getDataSize());
        delete[] act_z.getData();
    } else {
        Matrix predictions = softmax(z);
        Matrix labels = (*tensorMap)["lab"];

        float acc, loss;
        getTrainStat(predictions, labels, acc, loss);
        printLog(nodeId, "batch Acc: %f, Loss: %f", acc, loss);

        Matrix d_output = hadamardSub(predictions, labels);
        Matrix weight = msgService.getWeightMatrix(layer);
        Matrix interGrad = d_output.dot(weight, false, true);
        memcpy((*tensorMap)["grad"].getData(), interGrad.getData(),
               interGrad.getDataSize());

        Matrix ah = (*tensorMap)["ah"];
        Matrix weightUpdates = ah.dot(d_output, true, false);
        msgService.sendWeightUpdate(weightUpdates, layer);
        delete[] interGrad.getData();
        delete[] d_output.getData();
        delete[] predictions.getData();
    }
    delete[] z.getData();
}

void CPUComm::processBackward(unsigned layer) {
    Matrix weight = msgService.getWeightMatrix(layer);
    Matrix grad = (*tensorMap)["aTg"];
    Matrix z = (*tensorMap)["z"];

    Matrix actDeriv = activateDerivative(z);
    Matrix interGrad = grad * actDeriv;

    Matrix ah = (*tensorMap)["ah"];
    Matrix weightUpdates = ah.dot(interGrad, true, false);
    msgService.sendWeightUpdate(weightUpdates, layer);
    if (layer != 0) {
        Matrix resultGrad = interGrad.dot(weight, false, true);
        memcpy((*tensorMap)["grad"].getData(), resultGrad.getData(),
               resultGrad.getDataSize());
        delete[] resultGrad.getData();
    }

    delete[] actDeriv.getData();
    delete[] interGrad.getData();

    if (layer == 0) msgService.prefetchWeightsMatrix(totalLayers);
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

void CPUComm::sendShutdownMessage() {
    // Send kill message.
    msgService.terminateWeightServers(weightServerAddrs);
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
    // printLog(nodeId, "batch loss %f, batch acc %f", loss, acc);
}