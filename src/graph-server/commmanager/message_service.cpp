#include "message_service.hpp"
static void doNotFreeBuffer(void *data, void *hint) {}

void
MessageService::sendWeightUpdate(Matrix &matrix, unsigned layer) {
    if (wSndThread.joinable()) {
        wSndThread.join();
    }
    if (wReqThread.joinable()) {
        wReqThread.join();
    }

    wSndThread = std::thread(
    [&](Matrix matrix, unsigned layer) {
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
    }, matrix, layer);
}

MessageService::MessageService(unsigned wPort_, unsigned nodeId_):
    wctx(1),
    nodeId(nodeId_),
    wPort(wPort_),
    wsocktReady(0),
    confirm(5) {
    weightSocket = new zmq::socket_t(wctx, ZMQ_DEALER);
}

void MessageService::setUpWeightSocket(char *addr) {
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
void MessageService::prefetchWeightsMatrix(unsigned totalLayers) {
    if (wSndThread.joinable()) {
        wSndThread.join();
    }
    if (wReqThread.joinable()) {
        wReqThread.join();
    }
    if (infoThread.joinable()) {
        infoThread.join();
    }

    weights = std::vector<Matrix *>(totalLayers, 0);
    wReqThread = std::thread(
    [ &, totalLayers]() {
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
    });
}

Matrix MessageService::getWeightMatrix(unsigned layer) {
    if (wSndThread.joinable())
        wSndThread.join();
    if (wReqThread.joinable())
        wReqThread.join();
    return *weights.at(layer);
}

void MessageService::sendInfoMessage(unsigned numLambdas) {
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
MessageService::terminateWeightServers(std::vector<char *> &weightServerAddrs) {
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
MessageService::sendShutdownMessage(zmq::socket_t &weightsocket) {
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
