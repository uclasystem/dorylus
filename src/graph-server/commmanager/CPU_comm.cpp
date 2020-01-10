#include "CPU_comm.hpp"
#include <omp.h>

static void doNotFreeBuffer(void *data, void *hint) {
}

extern "C" ResourceComm *createComm(CommInfo &commInfo) {
    return new CPUComm(commInfo.nodeId, commInfo.numNodes, commInfo.dataserverPort,
                       commInfo.wServersFile, commInfo.weightserverPort, commInfo.totalLayers);
}

extern "C" void destroyComm(CPUComm *cpuComm) {
    delete cpuComm;
}

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
    msgService = MessageServiceCPU(wPort, nodeId);
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
                                unsigned numLocalVertices_, unsigned numFeats, unsigned numFeatsNext_) {
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
        // delete[] act_z.getData();
    } else {
        Matrix predictions = softmax(z);
        memcpy(actData, predictions.getData(), predictions.getDataSize());
        // delete[] predictions.getData();
    }
    // delete[] z.getData();
}


void CPUComm::setTrainValidationSplit(float trainPortion, unsigned numLocalVertices) {
    split = trainPortion;
};

// For backward-prop.
void CPUComm::newContextBackward(unsigned layer, FeatType *oldGradBuf, FeatType *newGradBuf, std::vector<Matrix> *savedTensors, FeatType *targetBuf,
                                 unsigned numLocalVertices, unsigned inFeatDim, unsigned outFeatDim, unsigned targetDim) {
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
        memcpy(newGradMatrix.getData(), interGrad.getData(), interGrad.getNumElemts());
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
        memcpy(newGradMatrix.getData(), resultGrad.getData(), resultGrad.getNumElemts());

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


void
MessageServiceCPU::sendWeightUpdate(Matrix &matrix, unsigned layer) {
    if (wSndThread.joinable()) {
        auto t1 = std::chrono::high_resolution_clock::now();
        wSndThread.join();
        auto t2 = std::chrono::high_resolution_clock::now();
        std::cout << "sendWeightUpdate wait "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
                  << " milliseconds\n";
    }
    if (wReqThread.joinable()) {
        auto t2 = std::chrono::high_resolution_clock::now();
        wReqThread.join();
        auto t3 = std::chrono::high_resolution_clock::now();
        std::cout << "sendWeightUpdate wait 2 "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count()
                  << " milliseconds\n";
    }

    wSndThread = std::thread(
    [&](Matrix matrix, unsigned layer) {
        auto t1 = std::chrono::high_resolution_clock::now();
        // Send push header.
        zmq::message_t header(HEADER_SIZE);
        populateHeader((char *) header.data(), OP::PUSH_BACKWARD, layer, matrix.getRows(),
                       matrix.getCols());
        weightSocket->send(header, ZMQ_SNDMORE);
        zmq::message_t updateMsg(matrix.getData(), matrix.getDataSize(), doNotFreeBuffer, NULL);
        weightSocket->send(updateMsg);

        // Wait for updates settled reply.
        zmq::message_t confirm;
        weightSocket->recv(&confirm);

        auto t2 = std::chrono::high_resolution_clock::now();
        std::cout << "sendWeightUpdate [NETWORK] "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
                  << " milliseconds\n";
    }, matrix, layer);
}

MessageServiceCPU::MessageServiceCPU(unsigned wPort_, unsigned nodeId_):
    wctx(1),
    nodeId(nodeId_),
    wPort(wPort_),
    wsocktReady(0),
    confirm(5) {
    weightSocket = new zmq::socket_t(wctx, ZMQ_DEALER);
}

void MessageServiceCPU::setUpWeightSocket(char *addr) {
    wsocktReady = 1;
    char ipc_addr[50];
    unsigned ipc_addr_len = strlen(ipc_addr);
    size_t identity_len = sizeof(unsigned) + ipc_addr_len;
    char identity[identity_len];
    memcpy(identity, (char *) &nodeId, sizeof(unsigned));
    memcpy(identity + sizeof(unsigned), ipc_addr, ipc_addr_len);
    weightSocket->setsockopt(ZMQ_IDENTITY, identity, identity_len);
    char whost_port[50];
    sprintf(whost_port, "tcp://%s:%u", addr, wPort);
    printf("connect to %s\n", whost_port);
    weightSocket->connect(whost_port);
}

// This retrieve all weights at the beginning
// TODO: This can be improved by making it layer-wise prefectching
void MessageServiceCPU::prefetchWeightsMatrix(unsigned totalLayers) {
    if (wSndThread.joinable()) {
        auto t1 = std::chrono::high_resolution_clock::now();
        wSndThread.join();
        auto t2 = std::chrono::high_resolution_clock::now();
        std::cout << "sendWeightUpdate wait "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
                  << " milliseconds\n";
    }
    if (wReqThread.joinable()) {
        auto t2 = std::chrono::high_resolution_clock::now();
        wReqThread.join();
        auto t3 = std::chrono::high_resolution_clock::now();
        std::cout << "requestWeightsMatrix Wait "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count()
                  << " milliseconds\n";
    }
    if (infoThread.joinable()) {
        auto t2 = std::chrono::high_resolution_clock::now();
        infoThread.join();
        auto t3 = std::chrono::high_resolution_clock::now();
        std::cout << "Info Wait "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count()
                  << " milliseconds\n";
    }

    weights = std::vector<Matrix *>(totalLayers, 0);
    wReqThread = std::thread(
    [ &, totalLayers]() {
        auto t1 = std::chrono::high_resolution_clock::now();
        if (wSndThread.joinable())
            wSndThread.join();

        for(unsigned i = 0; i < weights.size(); ++i) {
            if(weights[i] != NULL) {
                delete weights[i]->getData();
                delete weights[i];
            }
        }

        for(unsigned j = 0; j < totalLayers; ++j) {
            // Send pull request.
            zmq::message_t header(HEADER_SIZE);
            populateHeader((char *) header.data(), OP::PULL_FORWARD, j);
            weightSocket->send(header);
            // Listen on respond.
            zmq::message_t respHeader(HEADER_SIZE);
            weightSocket->recv(&respHeader);
            // Parse the respond.
            unsigned layerResp = parse<unsigned>((char *) respHeader.data(), 1);
            if ((int)layerResp == -1) {      // Failed.
                std::cerr << "[ ERROR ] No corresponding matrix chunk!" << std::endl;
                exit(1);
            } else {                    // Get matrices data.
                unsigned rows = parse<unsigned>((char *) respHeader.data(), 2);
                unsigned cols = parse<unsigned>((char *) respHeader.data(), 3);
                zmq::message_t wData(rows * cols * sizeof(FeatType));
                weightSocket->recv(&wData);
                FeatType *wBuffer = new FeatType[rows * cols];
                memcpy((char *)wBuffer, (char *)wData.data(), rows * cols * sizeof(FeatType));
                Matrix m(rows, cols, wBuffer);
                weights[j] = new Matrix(m.getRows(), m.getCols(), m.getData());
            }
        }
        auto t2 = std::chrono::high_resolution_clock::now();
        std::cout << "prefetchWeightsMatrix [NETWORK] CPU"
                  << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
                  << " milliseconds\n";

    });
}

Matrix MessageServiceCPU::getWeightMatrix(unsigned layer) {
    if (wSndThread.joinable())
        wSndThread.join();
    if (wReqThread.joinable())
        wReqThread.join();
    return *weights.at(layer);
}

void MessageServiceCPU::sendInfoMessage(unsigned numLambdas) {
    printf("sendInfoMessage CPU \n");
    infoThread = std::thread(
    [ &, numLambdas]() {
        zmq::message_t header(HEADER_SIZE);
        populateHeader((char *) header.data(), OP::INFO, numLambdas);
        weightSocket->send(header);
        // Wait for info received reply.
        zmq::message_t confirm;
        weightSocket->recv(&confirm);
    });
}

void
MessageServiceCPU::terminateWeightServers(std::vector<char *> &weightServerAddrs) {
    if(nodeId != 0)
        return;

    printLog(nodeId, "Node 0 is terminating all weightservers\n");

    for (unsigned i = 0; i < weightServerAddrs.size(); ++i) {
        zmq::socket_t ws = zmq::socket_t(wctx, ZMQ_DEALER);
        char identity[] = "coordx";
        ws.setsockopt(ZMQ_IDENTITY, identity, strlen(identity) + 1);
        char whost_port[50];
        sprintf(whost_port, "tcp://%s:%u", weightServerAddrs[i], wPort);
        printLog(nodeId, "[GPU]Shutting Down Weightserver %s \n", whost_port);
        ws.connect(whost_port);
        sendShutdownMessage(ws);
        ws.close();
    }
}

void
MessageServiceCPU::sendShutdownMessage(zmq::socket_t &weightsocket) {
    zmq::message_t header(HEADER_SIZE);
    populateHeader((char *) header.data(), OP::TERM);
    weightSocket->send(header);

    // Set receive timeou 1s property on this weightsocket, in case that a weightserver is dying too quickly that it's
    // confirm message it not sent from buffer yet. Using timeout here because shutdown is not a big deal.
    weightSocket->setsockopt(ZMQ_RCVTIMEO, 1000);

    // Wait for termination confirmed reply.
    zmq::message_t confirm;
    weightSocket->recv(&confirm);
}
