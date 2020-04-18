#include "CPU_comm.hpp"
#include <omp.h>

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

CPUComm::CPUComm(unsigned nodeId_, unsigned numNodes_, unsigned dataserverPort_, const std::string &wServersFile_, unsigned wPort_, unsigned totalLayers_):
    ResourceComm(),
    totalLayers(totalLayers_),
    wServersFile(wServersFile_),
    nodeId(nodeId_),
    numNodes(numNodes_),
    currLayer(0),
    dPort(dataserverPort_),
    wPort(wPort_),
    weightSocket(ctx, ZMQ_DEALER) {
    msgService = MessageService(wPort, nodeId);
    loadWeightServers(weightServerAddrs, wServersFile);
    msgService.setUpWeightSocket(weightServerAddrs.at(nodeId % weightServerAddrs.size()));

    //send INFO to weight server
    if(nodeId < weightServerAddrs.size()) {
        unsigned count = 0;
        for (size_t i = 0; i < numNodes; ++i) {
            if(i % weightServerAddrs.size() == nodeId)
                count += 1;
        }
        msgService.sendInfoMessage(count);
    }
    msgService.prefetchWeightsMatrix(totalLayers);
}


void CPUComm::newContextForward(unsigned layer, FeatType *dataBuf, FeatType *zData_, FeatType *actData_,
                                unsigned numLocalVertices_, unsigned numFeats, unsigned numFeatsNext_, bool pipeline) {
    // Create a new matrix object for workers to access.
    numLocalVertices = numLocalVertices_;
    currLayer = layer;
    actMatrix = Matrix(numLocalVertices_, numFeats, dataBuf);

    zData = zData_;
    actData = actData_;
    numFeatsNext = numFeatsNext_;
    printLog(nodeId, "CPU FORWARD context created.");
}

void CPUComm::requestForward(unsigned layer, bool lastLayer) {
    Matrix feats = actMatrix;
    Matrix weight = msgService.getWeightMatrix(layer);
    Matrix z = feats.dot(weight);
    memcpy(zData, z.getData(), z.getDataSize());

    if(!lastLayer) {
        Matrix act_z = activate(z); //z data get activated ...
        memcpy(actData, act_z.getData(), act_z.getDataSize());
        delete[] act_z.getData();
    } else {
        Matrix predictions = softmax(z);
        memcpy(actData, predictions.getData(), predictions.getDataSize());
        delete[] predictions.getData();
    }
    delete[] z.getData();
}


// For backward-prop.
void CPUComm::newContextBackward(unsigned layer, FeatType *oldGradBuf,
  FeatType *newGradBuf, std::vector<Matrix> *savedTensors, FeatType *targetBuf,
  unsigned numLocalVertices, unsigned inFeatDim, unsigned outFeatDim,
  unsigned targetDim, bool pipeline) {
    currLayer = layer;
    // Create new matrices object for workers to access.
    oldGradMatrix = Matrix(numLocalVertices, outFeatDim, oldGradBuf);
    newGradMatrix = Matrix(numLocalVertices, inFeatDim, newGradBuf);
    targetMatrix = Matrix(numLocalVertices, targetDim, targetBuf);
    this->savedTensors = savedTensors;
    printLog(nodeId, "CPU BACKWARD context created.");
}

void CPUComm::requestBackward(unsigned layer, bool lastLayer) {
    printLog(nodeId, "CPU BACKWARD request. %u", layer);
    if (lastLayer) {
        Matrix predictions = savedTensors[layer][TYPE::ACT - 1];
        Matrix labels = targetMatrix;
        Matrix d_output = hadamardSub(predictions, labels);
        Matrix weight = msgService.getWeightMatrix(layer);
        Matrix interGrad = d_output.dot(weight, false, true);
        memcpy(newGradMatrix.getData(), interGrad.getData(), interGrad.getDataSize());
        Matrix ah = savedTensors[layer][TYPE::AH - 1];
        Matrix weightUpdates = ah.dot(d_output, true, false);
        msgService.sendWeightUpdate(weightUpdates, layer);
        delete[] interGrad.getData();
        delete[] d_output.getData();

    } else {
        Matrix weight = msgService.getWeightMatrix(layer);
        Matrix grad = oldGradMatrix;
        Matrix z = savedTensors[layer][TYPE::Z - 1];
        Matrix actDeriv = activateDerivative(z);
        Matrix interGrad = grad * actDeriv;
        Matrix resultGrad = interGrad.dot(weight, false, true);
        Matrix ah = savedTensors[layer][TYPE::AH - 1];
        Matrix weightUpdates = ah.dot(interGrad, true, false);

        msgService.sendWeightUpdate(weightUpdates, layer);
        memcpy(newGradMatrix.getData(), resultGrad.getData(), resultGrad.getDataSize());

        delete[] actDeriv.getData();
        delete[] resultGrad.getData();
        delete[] interGrad.getData();

        if(layer == 0)
            msgService.prefetchWeightsMatrix(totalLayers);
    }
}


void CPUComm::sendShutdownMessage() {
    // Send kill message.
    msgService.terminateWeightServers(weightServerAddrs);
}
