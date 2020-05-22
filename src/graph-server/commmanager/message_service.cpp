#include "message_service.hpp"
static inline void populateHeader(void *header, unsigned op, Chunk &chunk) {
    char *ptr = (char *)header;
    memcpy(ptr, &op, sizeof(unsigned));
    memcpy(ptr + sizeof(unsigned), &chunk, sizeof(chunk));
}
static void doNotFreeBuffer(void *data, void *hint) {}

//-------------These are copied from yifan gcn------------
//-------------Search"MessageService" to Jump-------------
static void deleteMatrix(Matrix &mat) {
    if (!mat.empty()) {
        delete[] mat.getData();
        mat = Matrix();
    }
}

Matrix recvTensor(zmq::socket_t &socket) {
    zmq::message_t tensorHeader(TENSOR_HDR_SIZE);
    zmq::message_t tensorData;

    socket.recv(&tensorHeader);
    unsigned resp = parse<unsigned>((char *)tensorHeader.data(), 0);
    if (resp == ERR_HEADER_FIELD) {
        std::cerr << "Got error from server. Consult graph server output"
                  << std::endl;
        return Matrix();
    }
    std::string name = parseName((char *)tensorHeader.data());
    socket.recv(&tensorData);

    unsigned rows = parse<unsigned>((char *)tensorHeader.data(), 3);
    unsigned cols = parse<unsigned>((char *)tensorHeader.data(), 4);

    FeatType *data = new FeatType[rows * cols];
    std::memcpy(data, tensorData.data(), tensorData.size());

    return Matrix(name.c_str(), rows, cols, data);
}

std::vector<Matrix> reqTensors(zmq::socket_t &socket, Chunk &chunk,
                               std::vector<std::string> &tensorRequests) {
    bool empty = true;
    std::vector<Matrix> matrices;
    while (empty) {
        zmq::message_t header(HEADER_SIZE);
        populateHeader(header.data(), OP::PULL, chunk);
        socket.send(header, ZMQ_SNDMORE);
        unsigned numTensors = tensorRequests.size();
        for (unsigned u = 0; u < tensorRequests.size(); ++u) {
            std::string &name = tensorRequests[u];
            zmq::message_t tensorHeader(TENSOR_HDR_SIZE);
            populateHeader(tensorHeader.data(), chunk.localId, name.c_str());
            if (u < numTensors - 1) {
                socket.send(tensorHeader, ZMQ_SNDMORE);
            } else {
                socket.send(tensorHeader);
            }
        }

        unsigned more = 1;
        empty = false;
        while (more && !empty) {
            Matrix result = recvTensor(socket);
            if (result.empty()) {
                empty = result.empty();

                for (auto &M : matrices) deleteMatrix(M);
                matrices.clear();
                size_t usize = sizeof(more);
                socket.getsockopt(ZMQ_RCVMORE, &more, &usize);
            } else {
                matrices.push_back(result);

                size_t usize = sizeof(more);
                socket.getsockopt(ZMQ_RCVMORE, &more, &usize);
            }
        }
    }

    return matrices;
}

void sendTensors(zmq::socket_t &socket, Chunk &chunk,
                 std::vector<Matrix> &matrices, bool ack = false) {
    zmq::message_t header(HEADER_SIZE);
    populateHeader(header.data(), OP::PUSH, chunk);
    socket.send(header, ZMQ_SNDMORE);
    for (uint32_t u = 0; u < matrices.size(); ++u) {
        // std::cout << "Sending tensor " << matrices[u].name() << std::endl;
        zmq::message_t tensorHeader(TENSOR_HDR_SIZE);
        populateHeader(tensorHeader.data(), OP::PUSH,
                       matrices[u].name().c_str(), chunk.layer,
                       matrices[u].getRows(), matrices[u].getCols());
        zmq::message_t tensorData(matrices[u].getDataSize());
        std::memcpy(tensorData.data(), matrices[u].getData(),
                    matrices[u].getDataSize());

        socket.send(tensorHeader, ZMQ_SNDMORE);
        if (u < matrices.size() - 1) {
            socket.send(tensorData, ZMQ_SNDMORE);
        } else {
            socket.send(tensorData);
        }
    }

    if (ack) {
        std::cout << "Waiting on ACK" << std::endl;
        zmq::message_t ack;
        socket.recv(&ack);
        std::cout << "Received ACK" << std::endl;
    }
}

//-----------------------Finish Copy------------------------
MessageService::MessageService(unsigned wPort_, unsigned nodeId_)
    : wctx(1), nodeId(nodeId_), wPort(wPort_), wsocktReady(0), confirm(5), epoch(-1) {
    weightSocket = new zmq::socket_t(wctx, ZMQ_DEALER);
}

void MessageService::sendWeightUpdate(Matrix &matrix, unsigned layer) {
    if (wSndThread.joinable()) {
        wSndThread.join();
    }
    if (wReqThread.joinable()) {
        wReqThread.join();
    }

    wSndThread = std::thread(
        [&](Matrix matrix, unsigned layer) {
            matrix.setName("w");
            std::vector<Matrix> weightUpdates{matrix};
            Chunk c={0};
            c.layer = layer;
            c.epoch=epoch;
            c.globalId=nodeId;
            c.localId=nodeId;
            sendTensors(*weightSocket, c, weightUpdates);
            // delete[] matrix.getData();
        },
        matrix, layer);
}

void MessageService::setUpWeightSocket(char *addr) {
    wsocktReady = 1;
    char ipc_addr[50];
    unsigned ipc_addr_len = strlen(ipc_addr);
    size_t identity_len = sizeof(unsigned) + ipc_addr_len;
    char identity[identity_len];
    memcpy(identity, (char *)&nodeId, sizeof(unsigned));
    memcpy(identity + sizeof(unsigned), ipc_addr, ipc_addr_len);
    weightSocket->setsockopt(ZMQ_IDENTITY, identity, identity_len);
    char whost_port[50];
    sprintf(whost_port, "tcp://%s:%u", addr, wPort);
    // printf("connect to %s\n", whost_port);
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

    epoch++;
    weights = std::vector<Matrix *>(totalLayers, 0);
    wReqThread = std::thread([&, totalLayers]() {
        if (wSndThread.joinable()) wSndThread.join();

        for (unsigned i = 0; i < weights.size(); ++i) {
            if (weights[i] != NULL) {
                delete[] weights[i]->getData();
                delete weights[i];
            }
        }
        for (unsigned j = 0; j < totalLayers; ++j) {
            Chunk c={0};
            c.layer = j;
            c.epoch = epoch;
            c.globalId=nodeId;
            c.localId=nodeId;
            std::vector<std::string> weightRequests{"w"};
            Matrix m = reqTensors(*weightSocket, c, weightRequests)[0];
            weights[j] = new Matrix(m.getRows(), m.getCols(), m.getData());
        }
    });
}

Matrix MessageService::getWeightMatrix(unsigned layer) {
    if (wSndThread.joinable()) wSndThread.join();
    if (wReqThread.joinable()) wReqThread.join();
    return *weights.at(layer);
}

void MessageService::sendAccloss(float acc, float loss, unsigned vtcsCnt) {
    if (wSndThread.joinable()) {
        wSndThread.join();
    }

    acc *= vtcsCnt;
    loss *= vtcsCnt;
    Chunk chunk = { nodeId, nodeId, 0, vtcsCnt - 1, 1, PROP_TYPE::FORWARD, epoch, true };

    zmq::message_t header(HEADER_SIZE);
    populateHeader(header.data(), OP::EVAL, chunk);
    zmq::message_t payload(2 * sizeof(float));
    char *bufPtr = (char *)payload.data();
    memcpy(bufPtr, &acc, sizeof(float));
    bufPtr += sizeof(float);
    memcpy(bufPtr, &loss, sizeof(float));

    weightSocket->send(header, ZMQ_SNDMORE);
    weightSocket->send(payload);
}


// void MessageService::terminateWeightServers(
//     std::vector<char *> &weightServerAddrs) {
//     if (nodeId != 0) return;

//     printLog(nodeId, "Node 0 is terminating all weightservers");

//     for (unsigned i = 0; i < weightServerAddrs.size(); ++i) {
//         zmq::socket_t ws = zmq::socket_t(wctx, ZMQ_DEALER);
//         char identity[] = "coordx";
//         ws.setsockopt(ZMQ_IDENTITY, identity, strlen(identity) + 1);
//         char whost_port[50];
//         sprintf(whost_port, "tcp://%s:%u", weightServerAddrs[i], wPort);
//         printLog(nodeId, "[GPU]Shutting Down Weightserver %s", whost_port);
//         ws.connect(whost_port);
//         sendShutdownMessage(ws);
//         ws.close();
//     }
// }

// void MessageService::sendShutdownMessage(zmq::socket_t &weightsocket) {
//     zmq::message_t header(HEADER_SIZE);
//     populateHeader((char *)header.data(), OP::TERM);
//     weightSocket->send(header);

//     // Set receive timeou 1s property on this weightsocket, in case that a
//     // weightserver is dying too quickly that it's confirm message it not sent
//     // from buffer yet. Using timeout here because shutdown is not a big deal.
//     weightSocket->setsockopt(ZMQ_RCVTIMEO, 1000);

//     // Wait for termination confirmed reply.
//     zmq::message_t confirm;
//     weightSocket->recv(&confirm);
// }
