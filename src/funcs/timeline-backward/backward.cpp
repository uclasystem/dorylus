#include <aws/core/Aws.h>
#include <aws/core/utils/json/JsonSerializer.h>
#include <aws/lambda-runtime/runtime.h>
#include <cblas.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <ratio>
#include <sstream>
#include <string>
#include <thread>
#include <zmq.hpp>

#include "../../common/matrix.hpp"
#include "../../common/utils.hpp"

#define SLEEP_PERIOD 1000
#define TIMEOUT_PERIOD 500

#define SND_MORE true
#define NO_MORE false

using namespace Aws::Utils::Json;
using namespace aws::lambda_runtime;
using namespace std::chrono;

bool lastLayer = false;
int session = -1;
GPUTimers gt;

static Matrix requestTensor(zmq::socket_t &socket, OP op, unsigned partId,
                            TYPE type = TYPE::AH, unsigned layer = 0) {
    zmq::message_t header(HEADER_SIZE);
    populateHeader((char *)header.data(), op, partId, type, layer);
    socket.send(header);

    Timer reqTimer;
    reqTimer.start();

    zmq::message_t respHeader;
    //    socket.recv(&respHeader);
    while (!socket.recv(&respHeader, ZMQ_DONTWAIT)) {
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

    unsigned layerResp = parse<unsigned>((char *)respHeader.data(), 1);
    if (layerResp == -2) {
        std::cerr << "[ ERROR ] Discard execution." << std::endl;
        exit(0);
    } else if (layerResp == -1) {
        std::cerr << "[ ERROR ] No corresponding matrix" << std::endl;
        return Matrix();
    } else {
        unsigned rows = parse<unsigned>((char *)respHeader.data(), 2);
        unsigned cols = parse<unsigned>((char *)respHeader.data(), 3);

        zmq::message_t matxData(rows * cols * sizeof(FeatType));
        socket.recv(&matxData);

        FeatType *matxBuffer = new FeatType[rows * cols];
        std::memcpy(matxBuffer, matxData.data(), matxData.size());

        Matrix m(rows, cols, matxBuffer);
        return m;
    }
}

/**
 *
 * Send weight updates back to weightserver.
 *
 */
static void sendMatrix(Matrix &matrix, zmq::socket_t &socket, unsigned id) {
    // Send push header.
    zmq::message_t header(HEADER_SIZE);
    populateHeader((char *)header.data(), OP::PUSH_BACKWARD, id,
                   matrix.getRows(), matrix.getCols());
    socket.send(header, ZMQ_SNDMORE);

    zmq::message_t updateMsg(matrix.getDataSize());
    std::memcpy(updateMsg.data(), matrix.getData(), matrix.getDataSize());
    socket.send(updateMsg);

    Timer sndTimer;
    sndTimer.start();

    // Wait for updates settled reply.
    zmq::message_t confirm;
    //    socket.recv(&confirm);
    while (!socket.recv(&confirm, ZMQ_DONTWAIT)) {
        usleep(SLEEP_PERIOD);
        if (sndTimer.peek() > TIMEOUT_PERIOD) {
            // zmq::message_t _hdr(HEADER_SIZE);
            // populateHeader((char *) _hdr.data(), OP::PUSH_BACKWARD, id,
            // matrix.getRows(), matrix.getCols());
            zmq::message_t _hdr;
            _hdr.copy(&header);
            socket.send(_hdr, ZMQ_SNDMORE);
            // zmq::message_t _updMsg(matrix.getDataSize());
            // std::memcpy(_updMsg.data(), matrix.getData(),
            // matrix.getDataSize());
            zmq::message_t _updMsg;
            _updMsg.copy(&updateMsg);
            socket.send(_updMsg);
            sndTimer.start();
        }
    }
}

/**
 *
 * Softmax on a row-wise matrix, where each element softmax with respect to its
 * row.
 *
 */
static Matrix softmaxRows(Matrix &mat) {
    FeatType *res = new FeatType[mat.getNumElemts()];

    for (unsigned i = 0; i < mat.getRows(); ++i) {
        unsigned length = mat.getCols();
        FeatType *vecSrc = mat.getData() + i * length;
        FeatType *vecDst = res + i * length;

        FeatType denom = 0.;
        for (unsigned j = 0; j < length; ++j) {
            vecDst[j] = std::exp(vecSrc[j]);
            denom += vecDst[j];
        }
        for (unsigned j = 0; j < length; ++j) vecDst[j] /= denom;
    }

    return Matrix(mat.getRows(), mat.getCols(), res);
}

/**
 *
 * Apply derivative of the activation function on a matrix.
 *
 */
static Matrix activateDerivative(Matrix &mat) {
    FeatType *res = new FeatType[mat.getNumElemts()];
    FeatType *zData = mat.getData();

    for (unsigned i = 0; i < mat.getNumElemts(); ++i)
        res[i] = 1 - std::pow(std::tanh(zData[i]), 2);

    return Matrix(mat.getRows(), mat.getCols(), res);
}

static void gradLoss(zmq::socket_t &data_socket, zmq::socket_t &weight_socket,
                     unsigned id, unsigned layer) {
    Matrix predictions;
    Matrix labels;
    Matrix weights;
    Matrix ah;
    auto treq = gt.getTimer(std::string("LNREQ<-") + "Id" + std::to_string(id) +
                            "Ly" + std::to_string(layer));
    treq->start();

    std::cout << "< BACKWARD > Getting predictions" << std::endl;
    do {
        predictions =
            requestTensor(data_socket, OP::PULL_BACKWARD, id, TYPE::ACT, layer);
    } while (predictions.empty());

    std::cout << "< BACKWARD > Getting labels" << std::endl;
    do {
        labels =
            requestTensor(data_socket, OP::PULL_BACKWARD, id, TYPE::LAB, layer);
    } while (labels.empty());

    std::cout << "< BACKWARD > Requesting AH" << std::endl;
    do {
        ah = requestTensor(data_socket, OP::PULL_BACKWARD, id, TYPE::AH, layer);
    } while (ah.empty());

    std::cout << "< BACKWARD > Getting weights" << std::endl;
    do {
        weights = requestTensor(weight_socket, OP::PULL_BACKWARD, layer);
    } while (weights.empty());
    treq->stop();
    auto tcomp = gt.getTimer(std::string("LC<-") + "Id" + std::to_string(id) +
                             "Ly" + std::to_string(layer));
    tcomp->start();
    // derivative of softmax
    std::cout << "< BACKWARD > Calculating cross entropy" << std::endl;
    Matrix d_output = predictions - labels;
    delete[] predictions.getData();
    delete[] labels.getData();

    // d_out * W^T
    std::cout << "< BACKWARD > Computing gradient" << std::endl;
    Matrix interGrad = d_output.dot(weights, false, true);
    delete[] weights.getData();

    // AH^T * d_out
    std::cout << "< BACKWARD > Computing weight updates" << std::endl;
    Matrix weightUpdates = ah.dot(d_output, true, false);
    delete[] ah.getData();
    delete[] d_output.getData();

    tcomp->stop();

    auto tsend = gt.getTimer(std::string("LNSend<-") + "Id" +
                             std::to_string(id) + "Ly" + std::to_string(layer));
    tsend->start();
    std::thread wThrd([&] {
        std::cout << "< BACKWARD > Sending weight updates" << std::endl;
        sendMatrix(weightUpdates, weight_socket, layer);
        delete[] weightUpdates.getData();
    });

    std::cout << "< BACKWARD > Sending gradient to graph server" << std::endl;
    sendMatrix(interGrad, data_socket, id);
    tsend->stop();
    delete[] interGrad.getData();
    wThrd.join();
}

static void gradLayer(zmq::socket_t &data_socket, zmq::socket_t &weight_socket,
                      unsigned id, unsigned layer) {
    Matrix grad;
    Matrix z;
    Matrix weights;
    Matrix ah;
    auto treq = gt.getTimer(std::string("LNREQ<-") + "Id" + std::to_string(id) +
                            "Ly" + std::to_string(layer));
    treq->start();
    // REQUESTING ALL NEEDED TENSORS FOR COMPUTATION
    do {
        std::cout << "< BACKWARD > Requesting Z values" << std::endl;
        z = requestTensor(data_socket, OP::PULL_BACKWARD, id, TYPE::Z, layer);
    } while (z.empty());

    do {
        std::cout << "< BACKWARD > Requesting gradient from graph server"
                  << std::endl;
        grad = requestTensor(data_socket, OP::PULL_BACKWARD, id, TYPE::GRAD,
                             layer);
    } while (grad.empty());

    std::cout << "< BACKWARD > Requesting AH" << std::endl;
    do {
        ah = requestTensor(data_socket, OP::PULL_BACKWARD, id, TYPE::AH, layer);
    } while (ah.empty());

    std::cout << "< BACKWARD > Requesting weights" << std::endl;
    do {
        weights = requestTensor(weight_socket, OP::PULL_BACKWARD, layer);
    } while (weights.empty());
    // END REQUESTING ALL NEEDED TENSORS FOR COMPUTATION
    treq->stop();

    auto tcomp = gt.getTimer(std::string("LC<-") + "Id" + std::to_string(id) +
                             "Ly" + std::to_string(layer));
    tcomp->start();
    // BACKWARDS COMPUTATION
    std::cout << "< BACKWARD > Calculating derivative of activation "
              << z.shape() << std::endl;
    Matrix actDeriv = activateDerivative(z);
    delete[] z.getData();

    std::cout << "< BACKWARD > Hadamard multiplication" << grad.shape() << " "
              << actDeriv.shape() << std::endl;
    Matrix interGrad = grad * actDeriv;
    delete[] grad.getData();
    delete[] actDeriv.getData();

    std::cout << "< BACKWARD > MatMul(gradient, weights) " << interGrad.shape()
              << " " << weights.shape() << std::endl;
    Matrix resultGrad = interGrad.dot(weights, false, true);
    delete[] weights.getData();

    std::cout << "< BACKWARD > Computing weight updates " << ah.shape() << " "
              << interGrad.shape() << std::endl;
    Matrix weightUpdates = ah.dot(interGrad, true, false);
    delete[] ah.getData();
    delete[] interGrad.getData();
    // END BACKWARDS COMPUTATION
    tcomp->stop();

    // SENDING BACKWARDS RESULTS
    std::thread wThd([&] {
        std::cout << "< BACKWARD > Sending weight updates" << std::endl;
        sendMatrix(weightUpdates, weight_socket, layer);
        delete[] weightUpdates.getData();
    });

    auto tsend = gt.getTimer(std::string("LNSend<-") + "Id" +
                             std::to_string(id) + "Ly" + std::to_string(layer));
    tsend->start();
    std::cout << "< BACKWARD > Sending gradient to graph server" << std::endl;
    sendMatrix(resultGrad, data_socket, id);
    tsend->stop();
    delete[] resultGrad.getData();
    wThd.join();
    // END SENDING BACKWARDS RESULTS
}

/**
 *
 * Main logic:
 *
 *      1. Querying matrices chunks from dataserver;
 *      2. Querying weight matrices from weightserver;
 *      3. Conduct the gradient descent computation to get weight updates;
 *      4. Send weight updates back to weight servers.
 *
 */
static invocation_response backward_prop(std::string dataserver,
                                         std::string weightserver,
                                         unsigned dport, unsigned wport,
                                         unsigned id, unsigned layer,
                                         bool lastLayer) {
    zmq::context_t ctx(1);

    //
    // Lambda socket identity is set to:
    //
    //      [ 4 Bytes partId ] | [ n Bytes the string of dataserverIp ]
    //
    // One should extract the partition id by reading the first 4 Bytes, which
    // is simply parse<unsigned>(...).
    //
    size_t identity_len = sizeof(unsigned) * 3 + dataserver.length();
    char identity[identity_len];
    memcpy(identity, (char *)&id, sizeof(unsigned));
    std::srand(time(NULL));
    *(unsigned *)(identity + sizeof(unsigned)) = layer;
    *(unsigned *)(identity + sizeof(unsigned) * 2) = rand();
    memcpy(identity + sizeof(unsigned) * 3, (char *)dataserver.c_str(),
           dataserver.length());

    try {
        zmq::socket_t weights_socket(ctx, ZMQ_DEALER);
        weights_socket.setsockopt(ZMQ_IDENTITY, identity, identity_len);
        char whost_port[50];
        sprintf(whost_port, "tcp://%s:%u", weightserver.c_str(), wport);
        weights_socket.connect(whost_port);

        zmq::socket_t data_socket(ctx, ZMQ_DEALER);
        data_socket.setsockopt(ZMQ_IDENTITY, identity, identity_len);
        char dhost_port[50];
        sprintf(dhost_port, "tcp://%s:%u", dataserver.c_str(), dport);
        data_socket.connect(dhost_port);

        if (lastLayer) {
            std::cout << "< BACKWARD > Computing gradient from loss"
                      << std::endl;
            gradLoss(data_socket, weights_socket, id, layer);
        } else {
            std::cout << "< BACKWARD > Computing gradient for this layer"
                      << std::endl;
            gradLayer(data_socket, weights_socket, id, layer);
        }
    } catch (std::exception &ex) {
        JsonValue jsonResponse;
        jsonResponse.WithBool("success", false);
        jsonResponse.WithString("reason", ex.what());

        auto response = jsonResponse.View().WriteCompact();
        return invocation_response::failure(response, "application/json");
    }

    JsonValue jsonResponse;
    jsonResponse.WithBool("success", true);
    jsonResponse.WithInteger("type", PROP_TYPE::BACKWARD);
    jsonResponse.WithInteger("id", id);
    jsonResponse.WithInteger("session", session);
    jsonResponse.WithString("gtimers", gt.lambdaReport());

    std::cout << gt.lambdaReport() << std::endl;

    auto response = jsonResponse.View().WriteCompact();
    return invocation_response::success(response, "application/json");
}

/** Handler that hooks with lambda API. */
static invocation_response my_handler(invocation_request const &request) {
    JsonValue json(request.payload);
    auto v = json.View();
    gt = GPUTimers();
    
    std::string dataserver = v.GetString("dataserver");
    std::string weightserver = v.GetString("weightserver");
    unsigned dport = v.GetInteger("dport");
    unsigned wport = v.GetInteger("wport");
    unsigned layer = v.GetInteger("layer");
    unsigned chunkId = v.GetInteger("id");
    session = v.GetInteger("session");
    lastLayer = v.GetBool("lastLayer");

    std::cout << "[ACCEPTED] Thread " << chunkId << " is requested from "
              << dataserver << ":" << dport << ", BACKWARD on layer " << layer
              << std::endl;

    return backward_prop(dataserver, weightserver, dport, wport, chunkId, layer,
                         lastLayer);
}

int main(int argc, char *argv[]) {
    run_handler(my_handler);

    return 0;
}
