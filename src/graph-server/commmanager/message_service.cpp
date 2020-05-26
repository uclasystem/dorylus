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
        std::cout << "Sending tensor " << matrices[u].name() << std::endl;
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
    : wctx(1), wsocket(wctx, ZMQ_DEALER), wPort(wPort_), wsocktReady(0),
      nodeId(nodeId_), confirm(5), epoch(-1) {

    for (int i = 0; i < 2; i++) {
        weights.push_back(Matrix());
        as.push_back(Matrix());
    }
}

void MessageService::setUpWeightSocket(char *addr) {
    wsocktReady = 1;
    char ipc_addr[50];
    unsigned ipc_addr_len = strlen(ipc_addr);
    size_t identity_len = sizeof(unsigned) + ipc_addr_len;
    char identity[identity_len];
    memcpy(identity, (char *)&nodeId, sizeof(unsigned));
    memcpy(identity + sizeof(unsigned), ipc_addr, ipc_addr_len);
    wsocket.setsockopt(ZMQ_IDENTITY, identity, identity_len);
    char whost_port[50];
    sprintf(whost_port, "tcp://%s:%u", addr, wPort);
    printf("connect to %s\n", whost_port);
    wsocket.connect(whost_port);
}


Matrix MessageService::getWeightMatrix(unsigned layer) {
    if (!weights[layer].empty()) {
        return weights[layer];
    }

    Chunk c = { 0, nodeId, 0 ,0, layer,
                PROP_TYPE::FORWARD, epoch, true };

    std::vector<std::string> weightRequests{ "w" };
    std::vector<Matrix> mats = reqTensors(wsocket, c, weightRequests);
    if (mats.empty() || mats[0].empty()) {
        printLog(nodeId, "Got empty weights from weight server");
        return Matrix();
    }
    weights[layer] = mats[0];

    return weights[layer];
}

void MessageService::sendWeightUpdate(Matrix &matrix, unsigned layer) {
    matrix.setName("w");
    std::vector<Matrix> weightUpdates{matrix};
    Chunk c = { 0, nodeId, 0 ,0, layer,
                PROP_TYPE::BACKWARD, epoch, true };
    sendTensors(wsocket, c, weightUpdates);

    deleteMatrix(weights[layer]);
}

Matrix MessageService::getaMatrix(unsigned layer) {
    if (!as[layer].empty()) {
        return as[layer];
    }

    Chunk c = { 0, nodeId, 0 ,0, layer,
                PROP_TYPE::FORWARD, epoch, false };

    std::vector<std::string> weightRequests{ "a_i" };
    std::vector<Matrix> mats = reqTensors(wsocket, c, weightRequests);
    if (mats.empty() || mats[0].empty()) {
        printLog(nodeId, "Got empty a vector from weight server");
        return Matrix();
    }
    as[layer] = mats[0];

    return as[layer];
}

void MessageService::sendaUpdate(Matrix &matrix, unsigned layer) {
    matrix.setName("a_i");
    std::vector<Matrix> weightUpdates{matrix};
    Chunk c = { 0, nodeId, 0 ,0, layer,
                PROP_TYPE::BACKWARD, epoch, false };
    sendTensors(wsocket, c, weightUpdates);

    deleteMatrix(as[layer]);
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
    wReqThread = std::thread([&, totalLayers]() {
        if (wSndThread.joinable()) wSndThread.join();

        for (unsigned i = 0; i < weights.size(); ++i) {
            deleteMatrix(weights[i]);
            deleteMatrix(as[i]);
        }
        for (unsigned j = 0; j < totalLayers; ++j) {
            Chunk c = { 0 };
            c.layer = j;
            c.epoch = epoch;
            c.globalId=nodeId;
            c.localId=nodeId;
            std::vector<std::string> weightRequests{"w", "a_i"};
            std::vector<Matrix> wa = reqTensors(wsocket, c, weightRequests);
            weights[j] = wa[0];
            as[j] = wa[1];
        }
    });
}