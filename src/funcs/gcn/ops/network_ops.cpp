#include "network_ops.hpp"

Matrix recvTensor(zmq::socket_t& socket) {
    zmq::message_t tensorHeader(TENSOR_HDR_SIZE);
    zmq::message_t tensorData;

    socket.recv(&tensorHeader);
    unsigned resp = parse<unsigned>((char*)tensorHeader.data(), 0);
    if (resp == ERR_HEADER_FIELD) {
        std::cerr << "Got error from server. Consult graph server output" << std::endl;
        return Matrix();
    }
    std::string name = parseName((char*)tensorHeader.data());
    socket.recv(&tensorData);

    unsigned rows = parse<unsigned>((char*)tensorHeader.data(), 3);
    unsigned cols = parse<unsigned>((char*)tensorHeader.data(), 4);

    FeatType* data = new FeatType[rows * cols];
    std::memcpy(data, tensorData.data(), tensorData.size());

    return Matrix(name.c_str(), rows, cols, data);
}

std::vector<Matrix> reqTensors(zmq::socket_t& socket, Chunk &chunk,
                        std::vector<std::string>& tensorRequests) {
    bool empty = true;
    std::vector<Matrix> matrices;
    while (empty) {
        zmq::message_t header(HEADER_SIZE);
        populateHeader(header.data(), OP::PULL, chunk);
        socket.send(header, ZMQ_SNDMORE);
        unsigned numTensors = tensorRequests.size();
        for (unsigned u = 0; u < tensorRequests.size(); ++u) {
            std::string& name = tensorRequests[u];
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

                for (auto& M : matrices) deleteMatrix(M);
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

void sendTensors(zmq::socket_t& socket, Chunk &chunk,
            std::vector<Matrix>& matrices, bool ack) {
    zmq::message_t header(HEADER_SIZE);
    populateHeader(header.data(), OP::PUSH, chunk);
    socket.send(header, ZMQ_SNDMORE);
    for (uint32_t u = 0; u < matrices.size(); ++u) {
        std::cout << "Sending tensor " << matrices[u].name() << std::endl;
        zmq::message_t tensorHeader(TENSOR_HDR_SIZE);
        populateHeader(tensorHeader.data(), OP::PUSH, matrices[u].name().c_str(), chunk.layer,
          matrices[u].getRows(), matrices[u].getCols());
        zmq::message_t tensorData(matrices[u].getDataSize());
        std::memcpy(tensorData.data(), matrices[u].getData(), matrices[u].getDataSize());

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

/**
 *
 * Calculate batch loss and accuracy based on local forward predicts and labels.
 */
void sendAccLoss(zmq::socket_t &socket, Matrix &predicts, Matrix &labels, Chunk &chunk) {
    float acc = 0.0;
    float loss = 0.0;
    const unsigned vtcsCnt = chunk.upBound - chunk.lowBound;
    const unsigned featDim = labels.getCols();
    FeatType *currLabel = labels.getData();
    FeatType *currPred = predicts.getData();
    for (unsigned i = 0; i < vtcsCnt; i++) {
        acc += currLabel[argmax(currPred, currPred + featDim)];
        loss -= std::log(currPred[argmax(currLabel, currLabel + featDim)]);

        currLabel += featDim;
        currPred += featDim;
    }

    zmq::message_t header(HEADER_SIZE);
    populateHeader(header.data(), OP::EVAL, chunk);
    zmq::message_t payload(2 * sizeof(float));
    char *bufPtr = (char *)payload.data();
    memcpy(bufPtr, &acc, sizeof(float));
    bufPtr += sizeof(float);
    memcpy(bufPtr, &loss, sizeof(float));

    socket.send(header, ZMQ_SNDMORE);
    socket.send(payload);
}

void sendFinMsg(zmq::socket_t& socket, Chunk &chunk) {
    zmq::message_t header(HEADER_SIZE);
    populateHeader(header.data(), OP::FIN, chunk);
    socket.send(header);

    std::cout << "Waiting on ACK" << std::endl;
    zmq::message_t ack;
    socket.recv(&ack);
    std::cout << "Received ACK" << std::endl;
}
// end named-tensors
