#include "network_ops.hpp"

static Matrix
requestTensor(zmq::socket_t& socket, OP op, unsigned partId, TYPE type = TYPE::AH, unsigned layer = 0) {
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

static void
sendMatrices(Matrix& zResult, Matrix& actResult, zmq::socket_t& socket, unsigned id) {

    // Send push header.
    zmq::message_t header(HEADER_SIZE);
    populateHeader((char *) header.data(), OP::PUSH_FORWARD, id, zResult.getRows(), zResult.getCols());
    socket.send(header, ZMQ_SNDMORE);

    // Send zData and actData.
    zmq::message_t zData(zResult.getDataSize());
    std::memcpy(zData.data(), zResult.getData(), zResult.getDataSize());
    zmq::message_t actData(actResult.getDataSize());
    std::memcpy(actData.data(), actResult.getData(), actResult.getDataSize());
    socket.send(zData, ZMQ_SNDMORE);
    sendStart = timestamp_ms();
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
    sendEnd = timestamp_ms();
}

static void
sendMatrix(Matrix& matrix, zmq::socket_t& socket, unsigned id) {
    // Send push header.
    zmq::message_t header(HEADER_SIZE);
    populateHeader((char *) header.data(), OP::PUSH_BACKWARD, id, matrix.getRows(),
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
