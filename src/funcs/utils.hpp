#ifndef __LAMBDA_UTILS_HPP__
#define __LAMBDA_UTILS_HPP__

#include <iostream>
#include <zmq.hpp>
#include "../src/common/matrix.hpp"

/**
 *
 * Request the input matrix data from dataserver.
 *
 */
static Matrix
requestMatrix(zmq::socket_t& socket, OP op, unsigned id, bool data = false) {

    // Send pull request.
    zmq::message_t header(HEADER_SIZE);
    populateHeader((char *) header.data(), op, id);
    socket.send(header);

    // Listen on respond.
    zmq::message_t respHeader;
    socket.recv(&respHeader);

    // Parse the respond.
    unsigned layerResp = parse<unsigned>((char *) respHeader.data(), 1);
    if (layerResp == -1) {      // Failed.
        std::cerr << "[ ERROR ] No corresponding matrix chunk!" << std::endl;
        return Matrix();
    } else {                    // Get matrix data.
        unsigned rows = parse<unsigned>((char *) respheader.data(), 2);
        unsigned cols = parse<unsigned>((char *) respheader.data(), 3);
        // if talking to the graph servers, tell us if this is training or validation
        if (data) {
            unsigned eval = parse<unsigned>((char*) respheader.data(), 4);
            evaluate = bool(eval);
        }

        zmq::message_t matxdata(rows * cols * sizeof(FeatType));
        socket.recv(&matxdata);

        FeatType *matxbuffer = new FeatType[rows * cols];
        std::memcpy(matxbuffer, matxdata.data(), matxdata.size());

        matrix m(rows, cols, matxbuffer);
        return m;
    }
}


/**
 *
 * request the input matrix data from weightserver.
 *
 */
static matrix
requestWeightsMatrix(zmq::socket_t& socket, unsigned layer) {

    // send pull request.
    zmq::message_t header(header_size);
    populateheader((char *) header.data(), OP::PULL_VTX_FORWARD, layer);
    socket.send(header);

    // Listen on respond.
    zmq::message_t respHeader;
    socket.recv(&respHeader);

    // Parse the respond.
    unsigned layerResp = parse<unsigned>((char *) respHeader.data(), 1);
    if (layerResp == ERR_HEADER_FIELD) {    // Failed.
        std::cerr << "[ ERROR ] No corresponding matrix chunk!" << std::endl;
        return Matrix();
    } else {                    // Get matrices data.
        unsigned rows = parse<unsigned>((char *) respHeader.data(), 2);
        unsigned cols = parse<unsigned>((char *) respHeader.data(), 3);
        zmq::message_t matxData(rows * cols * sizeof(float));
        socket.recv(&matxData);

        FeatType *matxBuffer = new FeatType[rows * cols];
        std::memcpy(matxBuffer, matxData.data(), matxData.size());

        Matrix m(rows, cols, matxBuffer);
        return m;
    }
}


/**
 *
 * Send multiplied matrix result back to dataserver.
 *
 */
static void
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

    // Wait for data settled reply.
    zmq::message_t confirm;
    socket.recv(&confirm);
}


#endif // __LAMBDA_UTILS_HPP__