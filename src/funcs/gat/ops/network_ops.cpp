#include "network_ops.hpp"
#include <cmath>

int recvTensor(zmq::socket_t& socket, Matrix &mat) {
    zmq::message_t tensorHeader(TENSOR_HDR_SIZE);
    zmq::message_t tensorData;

    if (!socket.recv(&tensorHeader)) {
        return 0;
    }
    unsigned resp = parse<unsigned>((char*)tensorHeader.data(), 0);
    if (resp == ERR_HEADER_FIELD) {
        std::cerr << "Got error from server. Consult graph server output" << std::endl;
        return -1;
    }
    std::string name = parseName((char*)tensorHeader.data());

    if (!socket.recv(&tensorData)) {
        return 0;
    }

    unsigned rows = parse<unsigned>((char*)tensorHeader.data(), 3);
    unsigned cols = parse<unsigned>((char*)tensorHeader.data(), 4);

    FeatType* data = new FeatType[rows * cols];
    std::memcpy(data, tensorData.data(), tensorData.size());

    mat.setName(name.c_str());
    mat.setRows(rows);
    mat.setCols(cols);
    mat.setData(data);

    return 0;
}

std::vector<Matrix> reqTensors(zmq::socket_t& socket, Chunk &chunk,
                        std::vector<std::string>& tensorRequests) {

#define INIT_PERIOD (5 * 1000u) // 5ms
#define MAX_PERIOD (500 * 1000u)
#define EXP_FACTOR 1.5

    unsigned sleepPeriod = INIT_PERIOD;

    bool empty = true;
    std::vector<Matrix> matrices;
    while (true) {
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
            Matrix result;
            int ret = recvTensor(socket, result);
            if (ret == -1) {
                for (auto& M : matrices) deleteMatrix(M);
                matrices.clear();
                return matrices;
            }
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

        if (RESEND && empty) {
            usleep(sleepPeriod);
            sleepPeriod *= EXP_FACTOR;
            sleepPeriod = std::min(sleepPeriod, MAX_PERIOD);
        } else {
            break;
        }
    }

    return matrices;

#undef INIT_PERIOD
#undef MAX_PERIOD
#undef EXP_FACTOR
}

Matrix reqEdgeTensor(zmq::socket_t& socket, Chunk& chunk, std::string name) {
    zmq::message_t header(HEADER_SIZE);
    zmq::message_t tensorHeader(TENSOR_HDR_SIZE);
    populateHeader(header.data(), OP::PULLE, chunk);
    populateHeader(tensorHeader.data(), chunk.localId, name.c_str());
    socket.send(header, ZMQ_SNDMORE);
    socket.send(tensorHeader);

    // std::cout << "TENSR REQ HEADERS SENT" << std::endl;

    zmq::message_t respHeader(TENSOR_HDR_SIZE);
    zmq::message_t matxMsg;

    socket.recv(&respHeader);
    unsigned rows = parse<unsigned>((char*)respHeader.data(), 0);
    unsigned cols = parse<unsigned>((char*)respHeader.data(), 1);
    // std::cout << "RECVD TENSOR RESP HEADER" << std::endl;
    // std::cout << "ROWS " << rows << " COLS " << cols << std::endl;

    socket.recv(&matxMsg);
    // std::cout << "RECVD MATX MSG" << std::endl;
    FeatType* data = new FeatType[rows * cols];
    std::memcpy(data, matxMsg.data(), matxMsg.size());

    return Matrix(rows, cols, data);
}

EdgeInfo reqEdgeInfo(zmq::socket_t& socket, Chunk& chunk) {
    zmq::message_t header(HEADER_SIZE);
    populateHeader(header.data(), OP::PULLEINFO, chunk);
    socket.send(header);
    // std::cout << "INFO REQ HEADER SENT" << std::endl;

    zmq::message_t responseHeader(TENSOR_HDR_SIZE);

    EdgeInfo eTensor;
    socket.recv(&responseHeader);
    eTensor.numLvids = parse<unsigned>(responseHeader.data(), 0);
    eTensor.nChunkEdges = parse<unsigned>(responseHeader.data(), 1);
    // std::cout << "RECVD INFO RESP HEADER" << std::endl;

    if (eTensor.numLvids == NOT_FOUND_ERR_FIELD
      || eTensor.numLvids == DUPLICATE_REQ_ERR_FIELD
      || eTensor.numLvids == CHUNK_DNE_ERR) {
        std::cerr << "GOT ERROR" << std::endl;
        return eTensor;
    }

    // std::cout << "WAITING FOR MSG" << std::endl;
    zmq::message_t edgeChunkInfoMsg;
    socket.recv(&edgeChunkInfoMsg);
    eTensor.edgePtrs = new unsigned long long[edgeChunkInfoMsg.size() / sizeof(unsigned long long)];
    std::memcpy(eTensor.edgePtrs, edgeChunkInfoMsg.data(), edgeChunkInfoMsg.size());
    // std::cout << "EDGE CHUNK INFO SIZE " << edgeChunkInfoMsg.size() << std::endl;

    if (eTensor.edgePtrs == NULL) {
        std::cout << "NULL NULL NULL NULLETY NULL" << std::endl;
        eTensor.numLvids = ERR_HEADER_FIELD;
        eTensor.nChunkEdges = ERR_HEADER_FIELD;
        return eTensor;
    }

    return eTensor;
}

int sendEdgeTensors(zmq::socket_t& socket, Chunk &chunk,
            std::vector<Matrix>& matrices, bool ack) {
    zmq::message_t header(HEADER_SIZE);
    populateHeader(header.data(), OP::PUSHE, chunk);
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

    int ret = 0;
    if (ack) {
        std::cout << "Waiting on ACK" << std::endl;
        zmq::message_t ack;
        socket.recv(&ack);
        if (ack.size() == sizeof(int) * 3) {
            ret = *(int *)ack.data();
        }
        std::cout << "Received ACK" << std::endl;
    }
    return ret;
}
int sendTensors(zmq::socket_t& socket, Chunk &chunk,
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

    int ret = 0;
    if (ack) {
        std::cout << "Waiting on ACK" << std::endl;
        zmq::message_t ack;
        socket.recv(&ack);
        if (ack.size() == sizeof(int) * 3) {
            ret = *(int *)ack.data();
        }
        std::cout << "Received ACK" << std::endl;
    }
    return ret;
}

/**
 *
 * Calculate batch loss and accuracy based on local forward predicts and labels.
 */
void sendAccLoss(zmq::socket_t &dsocket, zmq::socket_t &wsocket, Matrix &predicts, Matrix &labels, Chunk &chunk) {
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

    // send accloss to graph server
    if (false) {
        zmq::message_t header(HEADER_SIZE);
        populateHeader(header.data(), OP::EVAL, chunk);
        zmq::message_t payload(2 * sizeof(float));
        char *bufPtr = (char *)payload.data();
        memcpy(bufPtr, &acc, sizeof(float));
        bufPtr += sizeof(float);
        memcpy(bufPtr, &loss, sizeof(float));

        dsocket.send(header, ZMQ_SNDMORE);
        dsocket.send(payload);
    }


    // send accloss to weight server
    if (true) {
        zmq::message_t header(HEADER_SIZE);
        populateHeader(header.data(), OP::EVAL, chunk);
        zmq::message_t payload(2 * sizeof(float));
        char *bufPtr = (char *)payload.data();
        memcpy(bufPtr, &acc, sizeof(float));
        bufPtr += sizeof(float);
        memcpy(bufPtr, &loss, sizeof(float));

        wsocket.send(header, ZMQ_SNDMORE);
        wsocket.send(payload);
    }
}

int sendFinMsg(zmq::socket_t& socket, Chunk &chunk) {
    zmq::message_t header(HEADER_SIZE);
    populateHeader(header.data(), OP::FIN, chunk);
    socket.send(header);

    int ret = 0;
    std::cout << "Waiting on ACK" << std::endl;
    zmq::message_t ack;
    socket.recv(&ack);
    if (ack.size() == sizeof(int) * 3) {
        ret = *(int *)ack.data();
    }
    std::cout << "Received ACK" << std::endl;

    return ret;
}
// end named-tensors
