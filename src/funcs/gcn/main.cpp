#include <algorithm>
#include <chrono>
#include <ratio>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <cmath>

#include <cblas.h>
#include <zmq.hpp>

#include <aws/lambda-runtime/runtime.h>
#include <aws/core/Aws.h>
#include <aws/core/utils/json/JsonSerializer.h>

#include "../../common/matrix.hpp"
#include "../../common/utils.hpp"

#include "ops/forward_ops.hpp"
#include "ops/backward_ops.hpp"
#include "ops/network_ops.hpp"


using namespace Aws::Utils::Json;
using namespace aws::lambda_runtime;
using namespace std::chrono;

bool lastLayer = false;


invocation_response
finalLayer(zmq::socket_t& data_socket, zmq::socket_t& weights_socket,
  unsigned partId, unsigned layer) {
//    try {
//        Matrix w, ah, z, predictions, labels;
//        std::thread commThread;
//
//        // Request weights matrix of the current layer.
//        commThread = std::thread([&] {     // Weight requests run in a separate thread.
//            do {
//                w = requestTensor(weights_socket, OP::PULL_FORWARD, layer);
//            } while (weights.empty());
//        });
//
//        // Request feature activation matrix of the current layer.
//        std::cout << "< FORWARD > Asking dataserver..." << std::endl;
//        do {
//            ah = requestTensor(data_socket, OP::PULL_FORWARD, partId);
//        } while (feats.empty());
//        std::cout << "< FORWARD > Got data from dataserver." << std::endl;
//
//        commThread.join();
//
//        if (weights.empty()) {
//            JsonValue jsonResponse;
//            jsonResponse.WithBool("success", false);
//            jsonResponse.WithString("reason", "Weights empty");
//
//            auto response = jsonResponse.View().WriteCompact();
//            return invocation_response::success(response, "application/json");
//        }
//        if (feats.empty()) {
//            JsonValue jsonResponse;
//            jsonResponse.WithBool("success", false);
//            jsonResponse.WithString("reason", "Features empty");
//
//            auto response = jsonResponse.View().WriteCompact();
//            return invocation_response::success(response, "appliation/json");
//        }
//
//        // Get labels for backward
//        commThread = std::thread([&] {
//            do {
//                labels = requestTensor(data_socket, OP::PULL_BACKWARD, partId, TYPE::LAB);
//            while (labels.empty());
//        });
//        // Multiplication.
//        z = ah.dot(w);
//        predictions = softmax(z);
//        delete[] z.getData();
//
//        commThread.join();
//        // d_o = softmax`(predictions)
//        Matrix d_out = predictions - labels;
//        delete[] predictions.getData();
//        delete[] labels.getData();
//
//        // d_o * W^T
//        Matrix interGrad = d_out.dot(w, false, true);
//        delete[] weights.getData();
//        sendMatrix(interGrad, data_socket, partId);
//
//        // AH^T * d_o
//        Matrix d_W = ah.dot(d_out, true, false);
//        delete[] ah.getData();
//        delete[] d_out.getData();
//    } catch(std::exception &ex) {
//        JsonValue jsonResponse;
//        jsonResponse.WithBool("success", false);
//        jsonResponse.WithString("reason", ex.what());
//
//        auto response = jsonResponse.View().WriteCompact();
//        return invocation_response::success(response, "application/json");
//    }
}

invocation_response
backwardLayer(zmq::socket_t& data_socket, zmq::socket_t& weight_socket,
  unsigned partId, unsigned layer) {
//    Matrix grad, z, ah, w;
//    std::commThread = std::thread([&]{
//        std::cout << "< BACKWARD > Requesting weights" << std::endl;
//        do {
//            w = requestTensor(weight_socket, OP::PULL_BACKWARD, layer);
//        } while (weights.empty());
//    });
//
//    do {
//        std::cout << "< BACKWARD > Requesting gradient from graph server" << std::endl;
//        grad = requestTensor(data_socket, OP::PULL_BACKWARD, id, TYPE::GRAD, layer);
//    } while (grad.empty());
//
//    do {
//        std::cout << "< BACKWARD > Requesting Z values" << std::endl;
//        z = requestTensor(data_socket, OP::PULL_BACKWARD, id, TYPE::Z, layer);
//    } while (z.empty());
//
//    std::cout << "< BACKWARD > Requesting AH" << std::endl;
//    do {
//        ah = requestTensor(data_socket, OP::PULL_BACKWARD, id, TYPE::AH, layer);
//    } while (ah.empty());
//    commThread.join();
//
//    // BACKWARDS COMPUTATION
//    std::cout << "< BACKWARD > Calculating derivative of activation "
//              << z.shape() << std::endl;
//    Matrix actDeriv = activateDerivative(z);
//    delete[] z.getData();
//
//    std::cout << "< BACKWARD > Hadamard multiplication" << grad.shape() << " "
//              << actDeriv.shape() << std::endl;
//    Matrix interGrad = grad * actDeriv;
//    delete[] grad.getData();
//    delete[] actDeriv.getData();
//
//    std::cout << "< BACKWARD > MatMul(gradient, weights) " << interGrad.shape() << " "
//              << weights.shape() << std::endl;
//    Matrix resultGrad = interGrad.dot(weights, false, true);
//    delete[] weights.getData();
//
//    std::cout << "< BACKWARD > Computing weight updates " << ah.shape() << " "
//              << interGrad.shape() << std::endl;
//    Matrix weightUpdates = ah.dot(interGrad, true, false);
//    delete[] ah.getData();
//    delete[] interGrad.getData();
//    // END BACKWARDS COMPUTATION
//
//    // SENDING BACKWARDS RESULTS
//    std::thread wThd([&] {
//        std::cout << "< BACKWARD > Sending weight updates" << std::endl;
//        sendMatrix(weightUpdates, weight_socket, layer);
//        delete[] weightUpdates.getData();
//    });
//
//    std::cout << "< BACKWARD > Sending gradient to graph server" << std::endl;
//    sendMatrix(resultGrad, data_socket, id);
//    delete[] resultGrad.getData();
//    wThd.join();
//    // END SENDING BACKWARDS RESULTS
}

invocation_response
forwardLayer(zmq::socket_t& data_socket, zmq::socket_t& weights_socket, unsigned partId,
  unsigned layer) {
    std::vector<std::string> dataRequests{"AH" + std::to_string(layer)};
    std::vector<std::string> weightRequests("W" + std::to_string(layer)};

    std::cout << "Req data Tensors" << std::endl;
    std::vector<Matrix> matrices = reqTensors(data_socket, partId, dataRequests);
    std::cout << "Req weights" << std::endl;
    std::vector<Matrix> weights = reqTensors(weights_socket, partId, weightRequests);

    for (auto& M : matrices) {
        if (M.empty()){
            std::cout << M.name() << " is empty" << std::endl;
            return constructResp(false, partId, M.name() + " is empty");
        }
    }

    for (auto& W : weights) {
        if (W.empty()){
            std::cout << W.name() << " is empty" << std::endl;
            return constructResp(false, partId, W.name() + " is empty");
        }
    }

    Matrix& AH = matrices[0];
    Matrix& W = weights[0];

    char name[8];
    Matrix Z = AH.dot(W);
    sprintf(name, "Z%u", layer);
    Z.setName(name);

    Matrix H_l = tanh(Z);
    sprintf(name, "H%u", layer);
    H_l.setName(name);

    std::vector<Matrix> toSend;
    toSend.push_back(Z);
    toSend.push_back(H_l);

    sendTensors(data_socket, partId, toSend);

    // Clean up data
    for (auto& M : matrices)
        deleteMatrix(M);
    for (auto& W : weights)
        deleteMatrix(W);
    for (auto& M : toSend)
        deleteMatrix(M);

    return constructResp(true, partId, "Finished forward layer");
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
 *      6. If evaluate is true, check the model precision
 *
 */
invocation_response
apply_phase(std::string dataserver, std::string weightserver, unsigned dport,
  unsigned wport, unsigned id, unsigned layer, unsigned prop_dir,
  bool lastLayer) {
    zmq::context_t ctx(1);

    // Creating identity
    size_t identity_len = sizeof(unsigned) * 3 + dataserver.length();
    char identity[identity_len];
    memcpy(identity, (char *) &id, sizeof(unsigned));
    std::srand(time(NULL));
    *(unsigned *)(identity + sizeof(unsigned)) = layer;
    *(unsigned *)(identity + sizeof(unsigned) * 2) = rand();
    memcpy(identity + sizeof(unsigned) * 3, (char *) dataserver.c_str(), dataserver.length());

    zmq::socket_t weights_socket(ctx, ZMQ_DEALER);
    zmq::socket_t data_socket(ctx, ZMQ_DEALER);
    try {
        weights_socket.setsockopt(ZMQ_IDENTITY, identity, identity_len);
        char whost_port[50];
        sprintf(whost_port, "tcp://%s:%u", weightserver.c_str(), wport);
        weights_socket.connect(whost_port);

        data_socket.setsockopt(ZMQ_IDENTITY, identity, identity_len);
        char dhost_port[50];
        sprintf(dhost_port, "tcp://%s:%u", dataserver.c_str(), dport);
        data_socket.connect(dhost_port);
    } catch(std::exception& ex) {
        return constructResp(false, id, ex.what());
    }

    if (prop_dir == PROP_TYPE::FORWARD && !lastLayer) {
        return forwardLayer(data_socket, weights_socket, id, layer);
    } else if (prop_dir == PROP_TYPE::FORWARD && lastLayer) {
        return finalLayer(data_socket, weights_socket, id, layer);
    } else if (prop_dir == PROP_TYPE::BACKWARD) {
        return backwardLayer(data_socket, weights_socket, id, layer);
    }

    return constructResp(false, id, "Didn't run any config");
}


/** Handler that hooks with lambda API. */
invocation_response
my_handler(invocation_request const& request) {
    JsonValue json(request.payload);
    auto v = json.View();

    std::string dataserver = v.GetString("dataserver");
    std::string weightserver = v.GetString("weightserver");
    unsigned dport = v.GetInteger("dport");
    unsigned wport = v.GetInteger("wport");
    unsigned layer = v.GetInteger("layer");
    unsigned chunkId = v.GetInteger("id");
    unsigned prop_type = v.GetInteger("prop_dir");
    lastLayer = v.GetBool("lastLayer");

    std::cout << "[ACCEPTED] Thread " << chunkId << " is requested from "
              << dataserver << ":" << dport << ", FORWARD layer " << layer
              << "." << std::endl;

    return apply_phase(dataserver, weightserver, dport, wport, chunkId, layer, prop_type, lastLayer);
}

int
main(int argc, char *argv[]) {
    run_handler(my_handler);

    return 0;
}
