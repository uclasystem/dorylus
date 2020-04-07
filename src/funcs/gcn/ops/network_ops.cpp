#include "network_ops.hpp"

Matrix
requestTensor(zmq::socket_t& socket, OP op, unsigned partId, TYPE type, unsigned layer) {
    zmq::message_t header(HEADER_SIZE);
    populateHeader((char*) header.data(), op, partId, type, layer);
    socket.send(header);

    Timer reqTimer;
    reqTimer.start();

    zmq::message_t respHeader;
//    socket.recv(&respHeader);
    while(!socket.recv(&respHeader, ZMQ_DONTWAIT)) {
        usleep(SLEEP_PERIOD);
        if (reqTimer.peek() > TIMEOUT_PERIOD) {
            // zmq::message_t _hdr(HEADER_SIZE);
            // populateHeader((char *) _hdr.data(), op, partId, type, layer);
            zmq::message_t _hdr;
            _hdr.copy(&header);
            socket.send(_hdr);
            reqTimer.start();
        }
    }

    unsigned layerResp = parse<unsigned>((char*) respHeader.data(), 1);
    if (layerResp == -2) {
        std::cerr << "[ ERROR ] Discard execution." << std::endl;
        exit(0);
    } else if (layerResp == -1) {
        std::cerr << "[ ERROR ] No corresponding matrix" << std::endl;
        return Matrix();
    } else {
        unsigned rows = parse<unsigned>((char*) respHeader.data(), 2);
        unsigned cols = parse<unsigned>((char *) respHeader.data(), 3);

        zmq::message_t matxData(rows * cols * sizeof(FeatType));
        socket.recv(&matxData);

        FeatType *matxBuffer = new FeatType[rows * cols];
        std::memcpy(matxBuffer, matxData.data(), matxData.size());

        Matrix m(rows, cols, matxBuffer);
        return m;
    }
}

void
sendMatrices(Matrix& zResult, Matrix& actResult, zmq::socket_t& socket, unsigned id) {
    // Send push header.
    zmq::message_t header(HEADER_SIZE);
    populateHeader((char *) header.data(), OP::PUSH_VTX_FORWARD, id, zResult.getRows(), zResult.getCols());
    socket.send(header, ZMQ_SNDMORE);

    // Send zData and actData.
    zmq::message_t zData(zResult.getDataSize());
    std::memcpy(zData.data(), zResult.getData(), zResult.getDataSize());
    zmq::message_t actData(actResult.getDataSize());
    std::memcpy(actData.data(), actResult.getData(), actResult.getDataSize());
    socket.send(zData, ZMQ_SNDMORE);
    socket.send(actData);

    Timer sndTimer;
    sndTimer.start();

    // Wait for data settled reply.
    zmq::message_t confirm;
//    socket.recv(&confirm);
    while(!socket.recv(&confirm, ZMQ_DONTWAIT)) {
        usleep(SLEEP_PERIOD);
        if (sndTimer.peek() > TIMEOUT_PERIOD) {
            zmq::message_t _hdr;
            _hdr.copy(&header);
            socket.send(header, ZMQ_SNDMORE);
            // zmq::message_t _updMsg(matrix.getDataSize());
            // std::memcpy(_updMsg.data(), matrix.getData(), matrix.getDataSize());
            zmq::message_t _zDt;
            _zDt.copy(&zData);
            socket.send(_zDt, ZMQ_SNDMORE);
            zmq::message_t _actDt;
            _actDt.copy(&actData);
            socket.send(_actDt);
            sndTimer.start();
        }
    }
}

void
sendMatrix(Matrix& matrix, zmq::socket_t& socket, unsigned id) {
    // Send push header.
    zmq::message_t header(HEADER_SIZE);
    populateHeader((char *) header.data(), OP::PUSH_VTX_BACKWARD, id, matrix.getRows(),
                   matrix.getCols());
    socket.send(header, ZMQ_SNDMORE);

    zmq::message_t updateMsg(matrix.getDataSize());
    std::memcpy(updateMsg.data(), matrix.getData(), matrix.getDataSize());
    socket.send(updateMsg);

    Timer sndTimer;
    sndTimer.start();

    // Wait for updates settled reply.
    zmq::message_t confirm;
//    socket.recv(&confirm);
    while(!socket.recv(&confirm, ZMQ_DONTWAIT)) {
        usleep(SLEEP_PERIOD);
        if (sndTimer.peek() > TIMEOUT_PERIOD) {
            // zmq::message_t _hdr(HEADER_SIZE);
            // populateHeader((char *) _hdr.data(), OP::PUSH_BACKWARD, id, matrix.getRows(), matrix.getCols());
            zmq::message_t _hdr;
            _hdr.copy(&header);
            socket.send(_hdr, ZMQ_SNDMORE);
            // zmq::message_t _updMsg(matrix.getDataSize());
            // std::memcpy(_updMsg.data(), matrix.getData(), matrix.getDataSize());
            zmq::message_t _updMsg;
            _updMsg.copy(&updateMsg);
            socket.send(_updMsg);
            sndTimer.start();
        }
    }
}


// named-tensors
Matrix recvTensor(zmq::socket_t& socket) {
    zmq::message_t tensorHeader(TENSOR_HDR_SIZE);
    zmq::message_t tensorData;

    socket.recv(&tensorHeader);
    unsigned resp = parse<unsigned>((char*)tensorHeader.data(), 0);
    std::string name = parseName((char*)tensorHeader.data());
    if (resp == ERR_HEADER_FIELD) {
        std::cerr << "Got error from server. Consult graph server output" << std::endl;
        return Matrix();
    }
    socket.recv(&tensorData);

    unsigned rows = parse<unsigned>((char*)tensorHeader.data(), 3);
    unsigned cols = parse<unsigned>((char*)tensorHeader.data(), 4);

    FeatType* data = new FeatType[rows * cols];
    std::memcpy(data, tensorData.data(), tensorData.size());

    return Matrix(name.c_str(), rows, cols, data);
}

std::vector<Matrix> reqTensors(zmq::socket_t& socket, unsigned arg1,
  unsigned arg2, std::vector<std::string>& tensorRequests) {
    bool empty = true;
    std::vector<Matrix> matrices;
    while (empty) {
        zmq::message_t header(HEADER_SIZE);
        populateHeader(header.data(), OP::PULL, arg1, arg2);
        socket.send(header, ZMQ_SNDMORE);
        unsigned numTensors = tensorRequests.size();
        for (unsigned u = 0; u < tensorRequests.size(); ++u) {
            std::string& name = tensorRequests[u];
            zmq::message_t tensorHeader(TENSOR_HDR_SIZE);
            populateHeader(tensorHeader.data(), arg1, name.c_str());
            if (u < numTensors-1) {
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
            } else {
                matrices.push_back(result);

                size_t usize = sizeof(more);
                socket.getsockopt(ZMQ_RCVMORE, &more, &usize);
            }
        }
    }

    return matrices;
}

void sendTensors(zmq::socket_t& socket, unsigned arg1, unsigned arg2, unsigned arg3,
  std::vector<Matrix>& matrices, bool ack) {
    zmq::message_t header(HEADER_SIZE);
    populateHeader(header.data(), OP::PUSH, arg1, arg2, arg3);
    socket.send(header, ZMQ_SNDMORE);
    for (uint32_t u = 0; u < matrices.size(); ++u) {
        std::cout << "Sending tensor " << matrices[u].name() << std::endl;
        zmq::message_t tensorHeader(TENSOR_HDR_SIZE);
        populateHeader(tensorHeader.data(), OP::PUSH, matrices[u].name().c_str(), arg2,
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

void sendFinishedMessage(zmq::socket_t& socket, unsigned arg1, unsigned arg2, unsigned arg3,
  bool ack) {
    zmq::message_t header(HEADER_SIZE);
    populateHeader(header.data(), OP::FIN, arg1, arg2, arg3);
    socket.send(header);

    if (ack) {
        std::cout << "Waiting on ACK" << std::endl;
        zmq::message_t ack;
        socket.recv(&ack);
        std::cout << "Received ACK" << std::endl;
    }
}
// end named-tensors
