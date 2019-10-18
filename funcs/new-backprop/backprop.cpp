#include <chrono>
#include <ratio>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <algorithm>
#include <cmath>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <cblas.h>
#include <zmq.hpp>
#include <aws/lambda-runtime/runtime.h>

#include "../../src/common/matrix.hpp"
#include "../../src/common/utils.hpp"

#define LEARNING_RATE (0.1)


#define SND_MORE true
#define NO_MORE false


using boost::property_tree::ptree;
using boost::property_tree::read_json;
using boost::property_tree::write_json;
using namespace aws::lambda_runtime;
using namespace std::chrono;

bool lastLayer = false;


static Matrix
requestMatrix(zmq::socket_t& socket, OP op, unsigned id) {
    zmq::message_t header(HEADER_SIZE);
    populateHeader((char*) header.data(), op, id);
    socket.send(header);

    zmq::message_t respHeader;
    socket.recv(&respHeader);

    unsigned layerResp = parse<unsigned>((char*) respHeader.data(), 1);
    if (layerResp == -1) {
        std::cerr << "[ ERROR ] No corresponding matrix" << std::endl;
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

static Matrix
requestWeights(zmq::socket_t& socket, OP op, unsigned layer) {
    zmq::message_t header(HEADER_SIZE);
    populateHeader((char*) header.data(), op, layer);
    socket.send(header);

    zmq::message_t respHeader;
    socket.recv(&respHeader);
    unsigned layerResp = parse<unsigned>((char*) respHeader.data(), 1);
    if (layerResp == -1) {
        std::cerr << "[ ERROR ] No corresponding matrix" << std::endl;
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

static Matrix
requestTensor(zmq::socket_t& socket, OP op, unsigned partId, TYPE type = 0, unsigned layer = 0) {
    zmq::message_t header(HEADER_SIZE);
    populateHeader((char*) header.data(), op, partId, type, layer);
    socket.send(header);

    zmq::message_t respHeader;
    socket.recv(&respHeader);
    unsigned layerResp = parse<unsigned>((char*) respHeader.data(), 1);
    if (layerResp == -1) {
        std::cerr << "[ ERROR ] No corresponding matrix" << std::endl;
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


/**
 *
 * Send weight updates back to weightserver.
 *
 */
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

    // Wait for updates settled reply.
    zmq::message_t confirm;
    socket.recv(&confirm);
}


static void
sendIntermediateGradient(Matrix& grad, zmq::socket_t& socket, unsigned id) {
    zmq::message_t header(HEADER_SIZE);
    populateHeader((char*) header.data(), OP::PUSH_BACKWARD, id, grad.getRows(),
                    grad.getCols());
    socket.send(header, ZMQ_SNDMORE);

    zmq::message_t gradMsg(grad.getDataSize());
    std::memcpy(gradMsg.data(), grad.getData(), grad.getDataSize());
    socket.send(gradMsg);

    // Wait for data received reply
    zmq::message_t confirm;
    socket.recv(&confirm);
}



/**
 *
 * Softmax on a row-wise matrix, where each element softmax with respect to its row.
 *
 */
static Matrix
softmaxRows(Matrix& mat) {
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
        for (unsigned j = 0; j < length; ++j)
            vecDst[j] /= denom;
    }

    return Matrix(mat.getRows(), mat.getCols(), res);
}



/**
 *
 * Apply derivative of the activation function on a matrix.
 *
 */
static Matrix
activateDerivate(Matrix& mat) {
    FeatType *res = new FeatType[mat.getNumElemts()];
    FeatType *zData = mat.getData();

    for (unsigned i = 0; i < mat.getNumElemts(); ++i)
        res[i] = 1 - std::pow(std::tanh(zData[i]), 2);

    return Matrix(mat.getRows(), mat.getCols(), res);
}

static void
gradLoss(zmq::socket_t& data_socket, zmq::socket_t& weight_socket, unsigned id, unsigned layer) {
    Matrix predictions = requestTensor(data_socket, OP::PULL_BACKWARD, id, TYPE::ACT, layer);
    Matrix labels = requestTensor(data_socket, OP::PULL_BACKWARD, id, TYPE::LAB, layer);
    std::cout << "< BACKWARD > Getting predictions and labels" << std::endl;

    // derivative of softmax
    std::cout << "< BACKWARD > Calculating cross entropy" << std::endl;
    Matrix d_output = predictions - labels;

    // d_out * W^T
    std::cout << "< BACKWARD > Getting weights" << std::endl;
    Matrix weights = requestTensor(weight_socket, OP::PULL_BACKWARD, layer);
    std::cout << "< BACKWARD > Computing gradient" << std::endl;
    Matrix interGrad = d_output.dot(weights, false, true);

    // AH^T * d_out
    std::cout << "< BACKWARD > Requesting AH" << std::endl;
    Matrix ah = requestTensor(data_socket, OP::PULL_BACKWARD, id, TYPE::AH, layer);
    std::cout << "< BACKWARD > Computing weight updates" << std::endl;
    Matrix weightUpdates = ah.dot(d_output, true, false, LEARNING_RATE);

    std::cout << "< BACKWARD > Sending weight updates" << std::endl;
    sendMatrix(weightUpdates, weight_socket, layer);
    std::cout << "< BACKWARD > Sending gradient to graph server" << std::endl;
    sendMatrix(interGrad, data_socket, id);
}

static void
gradLayer(zmq::socket_t& data_socket, zmq::socket_t& weight_socket, unsigned id, unsigned layer) {

    std::cout << "< BACKWARD > Requesting gradient from graph server" << std::endl;
    Matrix grad = requestTensor(data_socket, OP::PULL_BACKWARD, id, TYPE::GRAD, layer);
    std::cout << "< BACKWARD > Requesting Z values" << std::endl;
    Matrix z = requestTensor(data_socket, OP::PULL_BACKWARD, id, TYPE::Z, layer);

    std::cout << "< BACKWARD > Calculating derivative of activation" << std::endl;
    Matrix actDeriv = activateDerivative(z);
    std::cout << "< BACKWARD > Hadamard multiplication" << std::endl;
    Matrix interGrad = grad * actDeriv;

    std::cout << "< BACKWARD > Getting weights" << std::endl;
    Matrix weights = requestTensor(weight_socket, OP::PULL_BACKWARD, layer);
    std::cout << "< BACKWARD > MatMul(gradient, weights)" << std::endl;
    Matrix resultGrad = interGrad.dot(weights, false, true);

    std::cout << "< BACKWARD > Requesting AH" << std::endl;
    Matrix ah = requestTensor(data_socket, OP::PULL_BACKWARD, id, TYPE::AH, layer);
    std::cout << "< BACKWARD > Computing weight updates" << std::endl;
    Matrix weightUpdates = ah.dot(interGrad, true, false, LEARNING_RATE);

    std::cout << "< BACKWARD > Sending weight updates" << std::endl;
    sendMatrix(weightUpdates, weight_socket, layer);
    std::cout << "< BACKWARD > Sending gradient to graph server" << std::endl;
    sendMatrix(resultGrad, data_socket, id);
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
static invocation_response
backward_prop(std::string dataserver, std::string weightserver, std::string dport, std::string wport,
              unsigned id, unsigned layer) {
    zmq::context_t ctx(1);

    //
    // Lambda socket identity is set to:
    //
    //      [ 4 Bytes partId ] | [ n Bytes the string of dataserverIp ]
    //
    // One should extract the partition id by reading the first 4 Bytes, which is simply parse<unsigned>(...).
    //
    size_t identity_len = sizeof(unsigned) + dataserver.length();
    char identity[identity_len];
    memcpy(identity, (char *) &id, sizeof(unsigned));
    memcpy(identity + sizeof(unsigned), (char *) dataserver.c_str(), dataserver.length());

    Timer getWeightsTimer;
    Timer getFeatsTimer;
    Timer computationTimer;
    Timer sendResTimer;

    try {
        zmq::socket_t weights_socket(ctx, ZMQ_DEALER);
        weights_socket.setsockopt(ZMQ_IDENTITY, identity, identity_len);
        char whost_port[50];
        sprintf(whost_port, "tcp://%s:%s", weightserver.c_str(), wport.c_str());
        weights_socket.connect(whost_port);

        zmq::socket_t data_socket(ctx, ZMQ_DEALER);
        data_socket.setsockopt(ZMQ_IDENTITY, identity, identity_len);
        char dhost_port[50];
        sprintf(dhost_port, "tcp://%s:%s", dataserver.c_str(), dport.c_str());
        data_socket.connect(dhost_port);

        Matrix resultGradient;
        if (lastLayer) {
            std::cout << "< BACKWARD > Computing gradient from loss" << std::endl;
            resultGradient = gradLoss(data_socket, weights_socket, id, layer);
        } else {
            std::cout << "< BACKWARD > Computing gradient for this layer" << std::endl;
            resultGradient = gradLayer(data_socket, weights_socket, id, layer);
        }

    } catch(std::exception &ex) {
        return invocation_response::failure(ex.what(), "application/json");
    }

    // Couldn't parse JSON with AWS SDK from ptree.
    // For now creating a string with the times to be parsed on server.
    std::string res = "[ BACKWARD ] " + std::to_string(id) + ": " +
                      std::to_string(getWeightsTimer.getTime()) + " " +     \
                      std::to_string(getFeatsTimer.getTime())  + " " +      \
                      std::to_string(computationTimer.getTime()) + " " +    \
                      std::to_string(sendResTimer.getTime());

    return invocation_response::success(res, "application/json");
}


/** Handler that hooks with lambda API. */
static invocation_response
my_handler(invocation_request const& request) {
    ptree pt;
    std::istringstream is(request.payload);
    read_json(is, pt);

    std::string dataserver = pt.get<std::string>("dataserver");
    std::string weightserver = pt.get<std::string>("weightserver");
    std::string dport = pt.get<std::string>("dport");
    std::string wport = pt.get<std::string>("wport");
    unsigned layer = pt.get<int>("layer");
    unsigned chunkId = pt.get<int>("id");
    lastLayer = pt.get<bool>("lastLayer");

    std::cout << "[ACCEPTED] Thread " << chunkId << " is requested from " << dataserver << ":" << dport
              << ", BACKWARD on " << numLayers << " layers." << std::endl;

    return backward_prop(dataserver, weightserver, dport, wport, chunkId, layer, lastLayer);
}

int
main(int argc, char *argv[]) {
    run_handler(my_handler);
    return 0;
}
