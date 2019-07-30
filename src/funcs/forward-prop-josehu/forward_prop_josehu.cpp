#include <chrono>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <cmath>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <cblas.h>
#include <zmq.hpp>
#include <aws/lambda-runtime/runtime.h>
#include "../../utils/utils.h"


#define SND_MORE true
#define NO_MORE false


using boost::property_tree::ptree;
using boost::property_tree::read_json;
using boost::property_tree::write_json;
using namespace aws::lambda_runtime;
using namespace std::chrono;


/**
 *
 * Request the input matrix data from dataserver.
 * 
 */
Matrix
requestMatrix(zmq::socket_t& socket, int32_t id) {
    
    // Send pull request.
    zmq::message_t header(HEADER_SIZE);
    populateHeader((char *) header.data(), OP::PULL, id);
    socket.send(header);

    // Listen on respond.
    zmq::message_t respHeader;
    socket.recv(&respHeader);

    // Parse the respond.
    int32_t layerResp = parse<int32_t>((char *) respHeader.data(), 1);
    if (layerResp == -1) {      // Failed.
        std::cerr << "[ ERROR ] No corresponding matrix chunk!" << std::endl;
        return Matrix();
    } else {                    // Get matrix data.
        int32_t rows = parse<int32_t>((char *) respHeader.data(), 2);
        int32_t cols = parse<int32_t>((char *) respHeader.data(), 3);
        zmq::message_t matxData(rows * cols * sizeof(DTYPE));
        socket.recv(&matxData);

        char *matxBuffer = new char[matxData.size()];
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
void
sendMatrix(Matrix& response, int32_t resType, zmq::socket_t& socket, bool duplicate, int32_t id) {
    if (!duplicate) {
        zmq::message_t header(HEADER_SIZE);
        populateHeader((char *) header.data(), OP::PUSH, id, response.rows, response.cols);
        socket.send(header, ZMQ_SNDMORE);
    }

    zmq::message_t matxData(response.getDataSize());
    std::memcpy(matxData.data(), response.getData(), response.getDataSize());

    if (!duplicate)
        socket.send(matxData, ZMQ_SNDMORE);
    else
        socket.send(matxData);
}


/**
 *
 * Matrix multiplication function.
 * 
 */
Matrix
dot(Matrix& features, Matrix& weights) {
    int m = features.rows, k = features.cols, n = weights.cols;
    Matrix result(m, n);

    auto resultData = new DTYPE[m * n];
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0,
                features.getData(), k, weights.getData(), n, 0.0, resultData, n);

    result.setData(resultData);

    return result;
}


/**
 *
 * Apply activation function on a matrix.
 * 
 */
Matrix
activate(Matrix& mat) {
    DTYPE *activationData = new DTYPE[mat.rows * mat.cols];
    DTYPE *zData = mat.getData();
    
    for (int i = 0; i < mat.rows * mat.cols; ++i)
        activationData[i] = std::tanh(zData[i]);

    return Matrix(mat.rows, mat.cols, activationData);
}


/**
 *
 * Main logic:
 * 
 *      1. Querying matrix data from dataserver;
 *      2. Querying weight matrix from weightserver;
 *      3. Conduct the matrix multiplication to get Z matrix;
 *      4. Perform activation on Z to get Activated matrix;
 *      5. Send both matrices back to data server.
 * 
 */
invocation_response
matmul(std::string dataserver, std::string weightserver, std::string dport, std::string wport, int32_t id, int32_t layer) {
    zmq::context_t ctx(1);

    //
    // Lambda socket identity is set to:
    //
    //      [ 4 Bytes partId ] | [ n Bytes the string of dataserverIp ]
    //
    // One should extract the partition id by reading the first 4 Bytes, which is simply parse<int32_t>(...).
    //
    std::cout << sizeof(int32_t) << " Dataserver str: " << dataserver << " | length = " << dataserver.length() << " | strlen = " << strlen(dataserver.c_str()) << std::endl;
    char identity[sizeof(int32_t) + dataserver.length()];
    memcpy(&identity, (char *) &id, sizeof(int32_t));
    memcpy((&identity) + sizeof(int32_t), (char *) dataserver.c_str(), dataserver.length());

    Timer getWeightsTimer;
    Timer getFeatsTimer;
    Timer computationTimer;
    Timer activationTimer;
    Timer sendResTimer;

    try {

        // Request weights matrix.
        Matrix weights;
        std::thread t([&] {
            std::cout << "< matmul > Asking weightserver..." << std::endl;
            getWeightsTimer.start();
            zmq::socket_t weights_socket(ctx, ZMQ_DEALER);
            weights_socket.setsockopt(ZMQ_IDENTITY, identity, sizeof(int32_t));
            char whost_port[50];
            sprintf(whost_port, "tcp://%s:%s", weightserver.c_str(), wport.c_str());
            weights_socket.connect(whost_port);
            weights = requestMatrix(weights_socket, layer);
            std::cout << "< matmul > Got data from weightserver." << std::endl;
            getWeightsTimer.stop();
        });

        // Request feature matrix.
        getFeatsTimer.start();
        std::cout << "< matmul > Asking dataserver..." << std::endl;
        zmq::socket_t data_socket(ctx, ZMQ_DEALER);
        data_socket.setsockopt(ZMQ_IDENTITY, identity, sizeof(int32_t));
        char dhost_port[50];
        sprintf(dhost_port, "tcp://%s:%s", dataserver.c_str(), dport.c_str());
        data_socket.connect(dhost_port);
        Matrix feats = requestMatrix(data_socket, id);
        std::cout << "< matmul > Got data from dataserver." << std::endl;
        getFeatsTimer.stop();

        t.join();

        if (weights.empty())
            return invocation_response::failure("Weights could not be loaded", "application/json");
        if (feats.empty())
            return invocation_response::failure("No chunk corresponding to request", "appliation/json");

        // Multiplication.
        computationTimer.start();
        std::cout << "< matmul > Doing the dot multiplication..." << std::endl;
        Matrix z = dot(feats, weights);
        computationTimer.stop();

        // Activation.
        activationTimer.start();
        std::cout << "< matmul > Doing the activation..." << std::endl;
        Matrix activations = activate(z);
        activationTimer.stop();

        // Send back to dataserver.
        sendResTimer.start();
        std::cout << "< matmul > Sending results back..." << std::endl;
        sendMatrix(z, 0, data_socket, false, id);
        sendMatrix(activations, 1, data_socket, true, id);
        std::cout << "< matmul > Results sent." << std::endl;
        sendResTimer.stop();

    } catch(std::exception &ex) {
        return invocation_response::failure(ex.what(), "application/json");
    }

    // Couldn't parse JSON with AWS SDK from ptree.
    // For now creating a string with the times to be parsed on server.
    std::string res = std::to_string(id) + ": " + std::to_string(getWeightsTimer.getTime()) + " " + \
                      std::to_string(getFeatsTimer.getTime())  + " " +      \
                      std::to_string(computationTimer.getTime()) + " " +    \
                      std::to_string(activationTimer.getTime()) + " " +     \
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
    int32_t layer = pt.get<int32_t>("layer");
    int32_t chunkId = pt.get<int32_t>("id");

    std::cout << "Thread " << chunkId << " is requested from " << dataserver << ":" << dport
              << ", layer " << layer << "." << std::endl;

    return matmul(dataserver, weightserver, dport, wport, chunkId, layer);
}


int
main(int argc, char *argv[]) {
    run_handler(my_handler);
    
    return 0;
}
