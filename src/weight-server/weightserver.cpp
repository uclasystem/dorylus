#include "weightserver.hpp"
#include "serverworker.hpp"

/**
 *
 * Weightserver constructor & destructor.
 *
 */
WeightServer::WeightServer(std::string &wserverFile, std::string &myPrIpFile, std::string &gserverFile,
                           unsigned _listenerPort, unsigned _serverPort, unsigned _gport,
                           std::string &configFile, std::string &tmpFile,
                           bool _sync, float _targetAcc)
    : ctx(1), frontend(ctx, ZMQ_ROUTER), backend(ctx, ZMQ_DEALER), // gsocket(ctx, ZMQ_DEALER),
      listenerPort(_listenerPort), serverPort(_serverPort), gport(_gport),
      dataCtx(1), publisher(dataCtx, ZMQ_PUB), subscriber(dataCtx, ZMQ_SUB),
      numLambdas(0), term(false), adam(true), sync(_sync), targetAcc(_targetAcc) {

    std::vector<std::string> allNodeIps =
        parseNodeConfig(configFile, wserverFile, myPrIpFile, gserverFile);
    setupSockets();
    // Read the dsh file to get info about all weight server nodes.
    initWServerComm(allNodeIps);

    // Read in layer configurations and initialize weight matrices.
    initWeights();
    initAdamOpt(adam);

    createOutputFile(tmpFile);
}

WeightServer::~WeightServer() {
    freeAdamOpt();
    stopWorkers();
    freeWeights();
    closeSockets();
    closeOutputFile();
}

/**
 *
 * Start a bunch of worker threads and create a proxy through frontend
 * to backend.
 *
 */
void
WeightServer::run() {
    char host_port[50];
    sprintf(host_port, "tcp://*:%u", listenerPort);
    std::cout << "Binding weight server to " << host_port << "..." << std::endl;
    frontend.bind(host_port);
    backend.bind("inproc://backend");

    WeightServer &me = *this;
    for (int i = 0; i < NUM_LISTENERS; ++i) {
        workers.push_back(new ServerWorker(ctx, me, i));
        worker_threads.push_back(new std::thread(std::bind(&ServerWorker::work, workers[i])));
        worker_threads[i]->detach();
    }

    std::thread proxyThd([&] {
        try {
            zmq::proxy(static_cast<void *>(frontend), static_cast<void *>(backend), nullptr);
        } catch (std::exception &ex) { /** Context termintated. */ }
    });
    proxyThd.detach();

    // create receiver thread
    recvThd = new std::thread(std::bind(&WeightServer::receiver, this));
    recvThd->detach();
}

void WeightServer::stopWorkers() {
    // Delete workers.
    std::cout << "[SHUTDOWN] Deleting workers" << std::endl;
    for (int i = 0; i < NUM_LISTENERS; ++i) {
        delete worker_threads[i];
        delete workers[i];
    }
}

/**
 *
 * Apply the updates in queue.
 *
 */
void WeightServer::applyUpdate(unsigned layer, std::string& name) {
    Timer updateTimer;
    updateTimer.start();

    Matrix &updateMat = weightsStore[layer][name].localUpdMat;

    ackCntMtx.lock();
    ackCnt += numNode - 1;
    ackCntMtx.unlock();

    zmq::message_t header(UPD_HEADER_SIZE);
    fillHeader(header, -1, CTRL_MSG::DATA);
    zmq::message_t describer(TENSOR_NAME_SIZE + sizeof(unsigned));
    fillTensorDescriber(describer, name, layer);
    zmq::message_t updateDataMsg(updateMat.getDataSize());
    std::memcpy(updateDataMsg.data(), updateMat.getData(), updateMat.getDataSize());
    pubMtx.lock();
    publisher.send(header, ZMQ_SNDMORE);
    publisher.send(describer, ZMQ_SNDMORE);
    publisher.send(updateDataMsg);
    pubMtx.unlock();

    std::string checkInfo = weightsStore[layer][name].tryApplyUpdate(adamOpt, layer);
    if (nodeId == 0 && checkInfo != "") {
        serverLog(std::string("Local Layer ") + std::to_string(layer) + " " + checkInfo);
    }

    std::unique_lock<std::mutex> ul(ackCntMtx);
    ackCntCV.wait(ul, [&] { return ackCnt == 0; });
}

void WeightServer::receiver() {
    while (!term) {
        unsigned sender;
        unsigned topic;
        zmq::message_t header;
        subMtx.lock();
        subscriber.recv(&header);
        parseHeader(header, sender, topic);
        if (topic == CTRL_MSG::DATA) {
            zmq::message_t describer;
            zmq::message_t updMsg;
            subscriber.recv(&describer);
            subscriber.recv(&updMsg);
            subMtx.unlock();

            zmq::message_t ack(UPD_HEADER_SIZE);
            fillHeader(ack, sender, CTRL_MSG::ACK);
            pushoutMsg(ack);

            std::string name;
            unsigned layer;
            parseTensorDescriber(describer, name, layer);

            FeatType *updateData = (FeatType *)updMsg.data();
            std::string checkInfo;
            if (sync) {
                unsigned ghostUpdCnt = weightsStore[layer][name].ghostUpdate(updateData);
                if (ghostUpdCnt == numNode - 1) {
                    checkInfo = weightsStore[layer][name].tryApplyUpdate(adamOpt, layer);
                }
            } else {
                checkInfo = weightsStore[layer][name].tryApplyUpdate(adamOpt, layer, updateData);
            }
            if (nodeId == 0 && checkInfo != "") {
                serverLog(std::string("Ghost Layer ") + std::to_string(layer) + " " + checkInfo);
            }
        } else if (topic == CTRL_MSG::ACK) {
            subMtx.unlock();
            std::lock_guard<std::mutex> lg(ackCntMtx);
            ackCnt--;
            if (ackCnt == 0) {
                ackCntCV.notify_all();
            }
        } else if (topic == CTRL_MSG::ACCLOSS) {
            zmq::message_t accLossMsg(sizeof(AccLoss));
            subscriber.recv(&accLossMsg);
            subMtx.unlock();
            AccLoss accloss;
            memcpy(&accloss, accLossMsg.data(), sizeof(AccLoss));
            if (nodeId == 0) {
                updateGlobalAccLoss(sender, accloss);
            }
        }
    }
}


void WeightServer::updateLocalAccLoss(Chunk &chunk, float acc, float loss) {
    AccLoss accloss(chunk.epoch, chunk.upBound - chunk.lowBound, acc, loss);
    accMtx.lock();
    auto found = accLossTable.find(chunk.globalId);
    if (found != accLossTable.end()) {
        // if this chunk has already existed, just update the accloss
        found->second = accloss;
        accMtx.unlock();
        return;
    }
    accLossTable[chunk.globalId] = accloss;

    if (accLossTable.size() < numLambdas) {
        accMtx.unlock();
    } else {
        AccLoss alSum;
        for (auto &kv : accLossTable) {
            alSum.epoch = std::max(alSum.epoch, kv.second.epoch);
            alSum.vtcsCnt += kv.second.vtcsCnt;
            alSum.acc += kv.second.acc;
            alSum.loss += kv.second.loss;
        }
        accLossTable.clear();
        accMtx.unlock();

        if (nodeId == 0) {
            updateGlobalAccLoss(nodeId, alSum);
        } else {
            zmq::message_t header(UPD_HEADER_SIZE);
            fillHeader(header, 0, CTRL_MSG::ACCLOSS);
            zmq::message_t accLossMsg(sizeof(AccLoss));
            std::memcpy(accLossMsg.data(), &alSum, sizeof(AccLoss));
            pubMtx.lock();
            publisher.send(header, ZMQ_SNDMORE);
            publisher.send(accLossMsg);
            pubMtx.unlock();
        }
    }
}

void WeightServer::updateGlobalAccLoss(unsigned node, AccLoss &accloss) {
    accMtx.lock();

    auto found = wsAccTable.find(node);
    if (found != wsAccTable.end()) {
        // if partial sum has already existed, just update it.
        found->second = accloss;
        accMtx.unlock();
    } else {
        wsAccTable[node] = accloss;

        if (wsAccTable.size() < numNode) {
            accMtx.unlock();
        } else { // wsAccTable.size() >= numNode
            AccLoss alSum;
            for (auto &kv : wsAccTable) {
                alSum.epoch = std::max(alSum.epoch, kv.second.epoch);
                alSum.vtcsCnt += kv.second.vtcsCnt;
                alSum.acc += kv.second.acc;
                alSum.loss += kv.second.loss;
            }
            wsAccTable.clear();
            accMtx.unlock();

            alSum.acc /= alSum.vtcsCnt;
            alSum.loss /= alSum.vtcsCnt;

            char logBuf[512];
            sprintf(logBuf, "Epoch %u, acc: %.3f, loss: %.3f",
                alSum.epoch, alSum.acc, alSum.loss);
            std::string alLog(logBuf);
            serverLog(alLog);

            tryEarlyStop(alSum);
        }
    }
}

void WeightServer::tryEarlyStop(AccLoss &accloss) {
    if (accloss.acc >= targetAcc) {
        std::lock_guard<std::mutex> lg(storeMtx);

        for (auto &wtm : weightsStore) {
            for (auto &kv : wtm) {
                kv.second.stopUpdate();
            }
        }

        for (unsigned i = 0; i < gsockets.size(); ++i) {
            zmq::message_t header(HEADER_SIZE);
            populateHeader(header.data(), OP::TERM);
            gsockets[i].send(header);
        }
    }
}

void WeightServer::clearAccLoss() {
    std::lock_guard<std::mutex> lg(accMtx);

    accLossTable.clear();
    wsAccTable.clear();
}

/**
 *
 * Use dsh file to open sockets to other weight servers.
 *
 */
void
WeightServer::initWServerComm(std::vector<std::string> &allNodeIps) {
    if (nodeId == 0)
        serverLog("Initializing nodes...");

    std::string myIp = allNodeIps[nodeId];
    // Everyone needs to bind to a publisher socket.
    // Need to use the private IP because of port restrictions.
    char hostPort[50];
    sprintf(hostPort, "tcp://%s:%u", myIp.c_str(), serverPort);
    publisher.bind(hostPort);
    for (std::string &ipStr : allNodeIps) {
        if (ipStr != myIp) {
            sprintf(hostPort, "tcp://%s:%u", ipStr.c_str(), serverPort);
            subscriber.connect(hostPort);
        }
    }

    // Subscribe process.
    if (nodeId == 0) {
        unsigned remaining = numNode - 1;
        // Keep polling until all workers reponsd
        while (remaining > 0) {
            // Send msg 1.
            zmq::message_t outMsg1(UPD_HEADER_SIZE);
            fillHeader(outMsg1, -1, CTRL_MSG::MASTERUP);
            publisher.ksend(outMsg1);

            std::this_thread::sleep_for(std::chrono::milliseconds(500));

            zmq::message_t inMsg;
            while (subscriber.krecv(&inMsg, ZMQ_DONTWAIT)) {    // Wait on ALL msg 2.
                unsigned sender;
                unsigned inCtrlType;
                parseHeader(inMsg, sender, inCtrlType);
                if (inCtrlType == CTRL_MSG::WORKERUP) {
                    --remaining;
                }
            }
        }

        // Send msg 3 (init finished).
        zmq::message_t outMsg2(UPD_HEADER_SIZE);
        fillHeader(outMsg2, -1, CTRL_MSG::INITDONE);
        pushoutMsg(outMsg2);
    } else {
        unsigned sender;
        unsigned msgType;
        zmq::message_t inMsg;   // Recv msg 1.
        while (subscriber.recv(&inMsg)) {
            parseHeader(inMsg, sender, msgType);

            if (msgType == CTRL_MSG::MASTERUP)
                break;
        }

        // Send msg 2 (ack).
        zmq::message_t outMsg(UPD_HEADER_SIZE);
        fillHeader(outMsg, sender, CTRL_MSG::WORKERUP);
        pushoutMsg(outMsg);

        while (subscriber.recv(&inMsg)) {
            unsigned sender;
            unsigned doneMsg;
            parseHeader(inMsg, sender, doneMsg);
            if (doneMsg == CTRL_MSG::INITDONE)
                break;
        }
    }

    if (nodeId == 0)
        serverLog("All weight servers connected.");
}

std::vector<std::string> WeightServer::parseNodeConfig(std::string &configFile, std::string &wserverFile,
                                                    std::string &myPrIpFile, std::string &gserverFile) {
    // Read the layer config file. Each line is a number of features.
    {
        std::ifstream infile(configFile.c_str());
        if (!infile.good())
            fprintf(stderr, "[ERROR] Cannot open layer configuration file: %s [Reason: %s]\n", configFile.c_str(), std::strerror(errno));

        assert(infile.good());

        // Loop through each line.
        std::string line;
        while (!infile.eof()) {
            std::getline(infile, line);
            boost::algorithm::trim(line);

            if (line.length() > 0)
                dims.push_back(std::stoul(line));
        }

        // Assert there is at least one layer (input -> output).
        assert(dims.size() > 1);
    }

    std::vector<std::string> allNodeIps;
    {
        // Read IP files
        std::string myIp;

        std::ifstream ipFile(myPrIpFile);
        assert(ipFile.good());
        std::getline(ipFile, myIp);
        ipFile.close();

        std::ifstream dshFile(wserverFile);
        std::string line, masterIp;
        while (std::getline(dshFile, line)) {
            boost::algorithm::trim(line);
            if (line.length() > 0) {
                std::string ip = line.substr(line.find('@') + 1);

                // Set first node in file as master.
                if (ip == myIp) {
                    nodeId = allNodeIps.size();
                }

                // Even if this is not your IP, it is the master IP.
                if (allNodeIps.empty())
                    masterIp = ip;

                allNodeIps.push_back(ip);
            }
        }
        numNode = allNodeIps.size();
    }

    // read graph server ip file
    if (nodeId == 0) {
        std::ifstream ipFile(gserverFile);
        assert(ipFile.good());
        std::string line, masterIp;
        while (std::getline(ipFile, line)) {
            boost::algorithm::trim(line);
            if (line.length() > 0) {
                gserverIps.push_back(line);
                gsockets.push_back(zmq::socket_t(ctx, ZMQ_DEALER));
                char hostPort[50];
                sprintf(hostPort, "tcp://%s:%u", line.c_str(), gport);
                // serverLog(std::string("gserver: ") + hostPort);
                unsigned idtyLen = sizeof(unsigned) * 3 + line.size();
                char identity[idtyLen];
                char *idtPtr = identity;
                for (unsigned i = 0; i < 3; ++i) {
                    *(unsigned *)idtPtr = -1u;
                    idtPtr += sizeof(unsigned);
                }
                memcpy(idtPtr, line.c_str(), line.size());
                gsockets[gsockets.size() - 1].setsockopt(ZMQ_IDENTITY, identity, idtyLen);
                gsockets[gsockets.size() - 1].connect(hostPort);
            }
        }
        // // only master node's IP needed
        // std::getline(ipFile, gserverIp);
        ipFile.close();
    }

    return allNodeIps;
}


/**
 *
 * Read in layer configurations.
 *
 */
void
WeightServer::initWeights() {
    weightsStore.resize(dims.size() - 1);
    wMtxs.resize(dims.size() - 1);
    uMtxs.resize(dims.size() - 1);

    // If master node, initialize the weight matrices according to the layer config.
    if (nodeId == 0) {
        for (unsigned u = 0; u < weightsStore.size(); ++u) {
            // Hardcoding this to xavier init for now. Eventually need to make it
            // configurable
            Matrix w = xavierInitializer(dims[u], dims[u + 1]);
            weightsStore[u]["w"] = WeightTensor(w, &wMtxs[u]["w"], &uMtxs[u]["w"], sync);

            // Initialize layer biases
            // TODO:
            //  Make this configurable based on whether or not a bias matrix is requested
            //  for a NN module
            bool initBiasFlag = false;
            if (initBiasFlag) {
                Matrix b = initBias(dims[u + 1]);
                weightsStore[u]["b"] = WeightTensor(b, &wMtxs[u]["b"], &uMtxs[u]["b"], sync);
            }
        }

        for (unsigned u = 0; u < weightsStore.size(); ++u)
            serverLog("Layer " + std::to_string(u) + " - Weights: " + weightsStore[u]["w"].currMat().shape());
    }

    distributeWeights();
    setGhostUpdTot(numNode - 1);
}

/**
 *
 * Used for weights init when tanh or some other symmetric
 * activation is being used
 *
 */
Matrix
WeightServer::xavierInitializer(unsigned dim1, unsigned dim2) {
    std::default_random_engine dre(8888);
    std::uniform_real_distribution<float> dist(-1, 1);

    unsigned dataSize = dim1 * dim2;
    float *dptr = new float[dataSize];

    for (unsigned ui = 0; ui < dataSize; ++ui)
        dptr[ui] = dist(dre);

    float normFactor = std::sqrt(6.0 / (float (dim1 + dim2)));
    for (unsigned ui = 0; ui < dataSize; ++ui)
        dptr[ui] *= normFactor;
    return Matrix("w", dim1, dim2, dptr);
}

/**
 *
 * Used for weights init when the ReLU or some other asymmetric
 * activation function is used
 *
 */
Matrix
WeightServer::kaimingInitializer(unsigned dim1, unsigned dim2) {
    std::default_random_engine dre(8888);
    std::normal_distribution<float> dist(0, 1);

    unsigned dataSize = dim1 * dim2;
    FeatType *dptr = new FeatType[dataSize];

    for (unsigned ui = 0; ui < dataSize; ++ui)
        dptr[ui] = dist(dre);

    float normFactor = std::sqrt(2.0 / (float(dim1)));
    for (unsigned ui = 0; ui < dataSize; ++ui)
        dptr[ui] *= normFactor;

    return Matrix("w", dim1, dim2, dptr);
}

/**
 *
 * Randomly initialize weights
 *
 */
Matrix
WeightServer::randomInitializer(unsigned dim1, unsigned dim2,
                                   float lowerBound, float upperBound) {
    assert(lowerBound < upperBound);
    std::default_random_engine dre(8888);
    std::uniform_real_distribution<float> dist(lowerBound, upperBound);

    unsigned dataSize = dim1 * dim2;
    FeatType *dptr = new FeatType[dataSize];

    for (unsigned ui = 0; ui < dataSize; ++ui)
        dptr[ui] = dist(dre);

    return Matrix("w", dim1, dim2, dptr);
}

/**
 *
 * Initialize bias vectors to output size of layer
 *
 */
Matrix
WeightServer::initBias(unsigned dim, float initVal) {
    // TODO:
    //  Generalize implementation of tensors/matrices to include 3-D matrices and vectors
    FeatType *dptr = new FeatType[dim];

    for (unsigned ui = 0; ui < dim; ++ui)
        dptr[ui] = initVal;

    return Matrix("b", dim, 1, dptr);
}



/**
 *
 * Distribute the weight matrices from the master to the other weight servers
 *
 */
void
WeightServer::distributeWeights() {
    if (nodeId == 0) {
        // Master sends all the weight matrices to the worker nodes.
        pubMtx.lock();
        zmq::message_t header(UPD_HEADER_SIZE);
        fillHeader(header, -1, CTRL_MSG::DATA);
        publisher.send(header, ZMQ_SNDMORE);

        for (unsigned i = 0; i < weightsStore.size(); ++i) {
            Matrix &weights = weightsStore[i]["w"].currMat();

            zmq::message_t weightData(weights.getDataSize());
            std::memcpy((char *) weightData.data(), weights.getData(), weights.getDataSize());

            if (i == weightsStore.size() - 1)
                publisher.send(weightData);
            else
                publisher.send(weightData, ZMQ_SNDMORE);
        }
        pubMtx.unlock();

        // Get an ACK from every worker node that they have received the weights.
        zmq::message_t inMsg(UPD_HEADER_SIZE);
        int acksNeeded = numNode - 1;
        while (acksNeeded > 0) {
            subMtx.lock();
            subscriber.recv(&inMsg);
            subMtx.unlock();

            unsigned sender;
            unsigned msgType;
            parseHeader(inMsg, sender, msgType);
            if (msgType == CTRL_MSG::ACK) {
                acksNeeded--;
            }
        }
        // Worker code.
    } else {
        // Worker receives each weight matrix.
        unsigned sender;
        unsigned msgType;
        zmq::message_t header(UPD_HEADER_SIZE);
        subMtx.lock();
        subscriber.recv(&header);
        parseHeader(header, sender, msgType);

        unsigned layer = 0;
        int more = 0;
        do {
            zmq::message_t weightData;
            subscriber.recv(&weightData);

            char *matxData = new char[weightData.size()];
            std::memcpy(matxData, weightData.data(), weightData.size());

            Matrix w(dims[layer], dims[layer+1], (FeatType*)matxData);
            weightsStore[layer]["w"] = WeightTensor(w, &wMtxs[layer]["w"], &uMtxs[layer]["w"], sync);
            ++layer;

            size_t more_size = sizeof(more);
            subscriber.getsockopt(ZMQ_RCVMORE, &more, &more_size);
        } while (more);
        subMtx.unlock();

        // After all matrices have been received, alert the master.
        zmq::message_t ackMsg(UPD_HEADER_SIZE);
        fillHeader(ackMsg, sender, CTRL_MSG::ACK);
        pubMtx.lock();
        publisher.send(ackMsg);
        pubMtx.unlock();
    }

    if (nodeId == 0)
        serverLog("All nodes up to date.");
}

void WeightServer::setLocalUpdTot(unsigned localUpdTot) {
    std::lock_guard<std::mutex> lg(storeMtx);

    numLambdas = localUpdTot;
    for (auto &wtm : weightsStore) {
        for (auto &kv : wtm) {
            kv.second.setLocalUpdTot(localUpdTot);
        }
    }
}

void WeightServer::setGhostUpdTot(unsigned ghostUpdTot) {
    for (auto &wtm : weightsStore) {
        for (auto &kv : wtm) {
            kv.second.setGhostUpdTot(ghostUpdTot);
        }
    }
}

void WeightServer::fillHeader(zmq::message_t &header, unsigned receiver, unsigned topic) {
    char *msgPtr = (char *)header.data();
    if (receiver == -1u) {
        sprintf(msgPtr, "FFFF");
    } else {
        sprintf(msgPtr, "%4X", receiver);
    }
    msgPtr += IDENTITY_SIZE;
    *((unsigned *)msgPtr) = nodeId;
    msgPtr += sizeof(unsigned);
    *((unsigned *)msgPtr) = topic;
    msgPtr += sizeof(unsigned);
}

void WeightServer::parseHeader(zmq::message_t &header, unsigned &sender, unsigned &topic) {
    char *msgPtr = (char *)header.data();
    msgPtr += IDENTITY_SIZE;
    sender = *(unsigned *)msgPtr;
    msgPtr += sizeof(unsigned);
    topic = *(unsigned *)msgPtr;
}

void WeightServer::fillTensorDescriber(zmq::message_t &td, std::string &name, unsigned layer) {
    char *msgPtr = (char *)td.data();
    sprintf(msgPtr, "%s", name.c_str());
    msgPtr += TENSOR_NAME_SIZE;
    *(unsigned *)msgPtr = layer;
}

void WeightServer::parseTensorDescriber(zmq::message_t &td, std::string &name, unsigned &layer) {
    char *msgPtr = (char *)td.data();
    name = std::string(msgPtr);
    msgPtr += TENSOR_NAME_SIZE;
    layer = *(unsigned *)msgPtr;
}

void WeightServer::pushoutMsg(zmq::message_t &msg) {
    std::lock_guard<std::mutex> lg(pubMtx);
    publisher.send(msg);
}

void WeightServer::pushoutMsgs(std::vector<zmq::message_t *> &msgs) {
    std::lock_guard<std::mutex> lg(pubMtx);
    unsigned cnt = msgs.size();
    for (unsigned u = 0; u < cnt - 1; u++) {
        publisher.send(*msgs[u], ZMQ_SNDMORE);
    }
    publisher.send(*msgs[cnt - 1]);
}

void WeightServer::setupSockets() {
    frontend.setsockopt(ZMQ_BACKLOG, 1000);
    backend.setsockopt(ZMQ_BACKLOG, 1000);

    publisher.setsockopt(ZMQ_SNDHWM, 0);    // Set no limit on message queue.
    publisher.setsockopt(ZMQ_RCVHWM, 0);

    subscriber.setsockopt(ZMQ_SNDHWM, 0);
    subscriber.setsockopt(ZMQ_RCVHWM, 0);
    char tag[IDENTITY_SIZE + 1];
    sprintf(tag, "%4X", nodeId);
    subscriber.setsockopt(ZMQ_SUBSCRIBE, tag, IDENTITY_SIZE);
    subscriber.setsockopt(ZMQ_SUBSCRIBE, "FFFF", IDENTITY_SIZE);
}

void WeightServer::closeSockets() {
    std::cout << "[SHUTDOWN] Closing ZMQ" << std::endl;

    for (auto &gs: gsockets) {
        gs.setsockopt(ZMQ_LINGER, 0);
        gs.close();
    }
    frontend.setsockopt(ZMQ_LINGER, 0);
    frontend.close();
    backend.setsockopt(ZMQ_LINGER, 0);
    backend.close();
    ctx.close();

    publisher.setsockopt(ZMQ_LINGER, 0);
    publisher.close();
    subscriber.setsockopt(ZMQ_LINGER, 0);
    subscriber.close();
    dataCtx.close();
}

void WeightServer::freeWeights() {
    unsigned numLayers = weightsStore.size();
    for (unsigned i = 0; i < numLayers; ++i) {
        for (auto &kv : weightsStore[i]) {
            kv.second.free();
        }
    }
}

void WeightServer::initAdamOpt(bool adam) {
    // Initialize the adam optimizer if this is the master
    if (adam) {
        adamOpt = new AdamOptimizer(LEARNING_RATE, dims);
    } else {
        adamOpt = NULL;
    }
}

void WeightServer::freeAdamOpt() {
    if (adamOpt) {
        delete adamOpt;
        adamOpt = NULL;
    }
}

void WeightServer::createOutputFile(std::string &fileName) {
    // Set output file name.
    fileName += std::to_string(nodeId);
    outfile.open(fileName, std::fstream::out);
    assert(outfile.good());
}

void WeightServer::closeOutputFile() {
    outfile.close();
}

/** Logging utility. */
void
WeightServer::serverLog(std::string info) {
    char msg[512];
    sprintf(msg, "[ WS %3d ] %s\n", nodeId, info.c_str());
    fprintf(stderr, "%s", msg);
}
