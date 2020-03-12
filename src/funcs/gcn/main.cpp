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

#include "utils.hpp"
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
    std::cout << "FINAL LAYER" << std::endl;
    std::vector<std::string> dataRequests{"AH" + std::to_string(layer), "LAB"};
    std::vector<std::string> weightRequests{"W" + std::to_string(layer)};

    std::vector<Matrix> matrices = reqTensors(data_socket, partId, layer, dataRequests);
    std::vector<Matrix> weights = reqTensors(weights_socket, partId, layer, weightRequests);

    if (matrices.empty() || weights.empty()) {
        return constructResp(false, partId, "Got error message from server");
    }

    for (auto& M : matrices) {
        if (M.empty()){
            std::cout << M.name() << " is empty" << std::endl;
            return constructResp(false, partId, M.name() + " is empty");
        } else {
            std::cout << "GOT " << M.name() << std::endl;
        }
    }
    for (auto& W : weights) {
        if (W.empty()){
            std::cout << W.name() << " is empty" << std::endl;
            return constructResp(false, partId, W.name() + " is empty");
        } else {
            std::cout << "GOT " << W.name() << std::endl;
        }
    }

    // Forward layer
    Matrix& AH = matrices[0];
    Matrix& W = weights[0];

    Matrix Z = AH.dot(W);
    Matrix preds = softmax(Z);
    deleteMatrix(Z);
    Matrix& labels = matrices[1];

    // Backward computation
    Matrix d_out = preds - labels;
    deleteMatrix(labels);
    deleteMatrix(preds);

    Matrix interGrad = d_out.dot(W, false, true);
    deleteMatrix(W);
    Matrix d_weights = AH.dot(d_out, true, false);
    deleteMatrix(AH);
    deleteMatrix(d_out);
    char name[8];
    sprintf(name, "GRAD%u", layer);
    interGrad.setName(name);

    sprintf(name, "W%u", layer);
    d_weights.setName(name);

    std::vector<Matrix> weightUpdates{d_weights};
    sendTensors(weights_socket, layer, layer, weightUpdates);

    std::vector<Matrix> toSend{interGrad};
    sendTensors(data_socket, partId, layer, toSend, true);

    std::cout << "SENT tensors and weight updates" << std::endl;

    // Clean up data
    for (auto& M : toSend)
        deleteMatrix(M);
    for (auto& M : weightUpdates)
        deleteMatrix(M);
    std::cout << "Data cleaned up" << std::endl;

    return constructResp(true, partId, "Finished final layer");
}

invocation_response
backwardLayer(zmq::socket_t& data_socket, zmq::socket_t& weights_socket,
  unsigned partId, unsigned layer) {
    std::cout << "BACKWARD LAYER" << std::endl;
    std::vector<std::string> dataReqs{"AH" + std::to_string(layer), "Z" + std::to_string(layer), "BAH" + std::to_string(layer+1)};
    std::vector<std::string> weightReqs{"W" + std::to_string(layer)};
    std::vector<Matrix> matrices = reqTensors(data_socket, partId, layer, dataReqs);
    std::vector<Matrix> weights = reqTensors(weights_socket, partId, layer, weightReqs);

    if (matrices.empty() || weights.empty()) {
        return constructResp(false, partId, "Got error message from server");
    }

    for (auto& M : matrices) {
        if (M.empty()){
            std::cout << M.name() << " is empty" << std::endl;
            return constructResp(false, partId, M.name() + " is empty");
        } else {
            std::cout << "GOT " << M.name() << std::endl;
        }
    }
    for (auto& W : weights) {
        if (W.empty()){
            std::cout << W.name() << " is empty" << std::endl;
            return constructResp(false, partId, W.name() + " is empty");
        } else {
            std::cout << "GOT " << W.name() << std::endl;
        }
    }

    Matrix& AH = matrices[0];
    Matrix& Z = matrices[1];
    Matrix& grad = matrices[2];

    Matrix& W = weights[0];
    char name[8];

    Matrix actDeriv = tanhDerivative(Z);
    deleteMatrix(Z);
    sprintf(name, "Z'");
    actDeriv.setName(name);

    Matrix interGrad = grad * actDeriv;
    deleteMatrix(grad);
    deleteMatrix(actDeriv);
    sprintf(name, "d_z");
    interGrad.setName(name);

    Matrix resultGrad = interGrad.dot(W, false, true);
    deleteMatrix(W);

    Matrix d_weights = AH.dot(interGrad, true, false);
    deleteMatrix(AH);
    deleteMatrix(interGrad);

    sprintf(name, "W%u", layer);
    d_weights.setName(name);

    sprintf(name, "GRAD%u", layer);
    resultGrad.setName(name);

    std::vector<Matrix> weightUpdates{d_weights};
    std::vector<Matrix> toSend{resultGrad};

    sendTensors(weights_socket, layer, layer, weightUpdates);
    sendTensors(data_socket, partId, layer, toSend, true);

    for (auto& M : toSend)
        deleteMatrix(M);
    for (auto& M : weightUpdates)
        deleteMatrix(M);

    return constructResp(true, partId, "Finished backward layer");
}

invocation_response
forwardLayer(zmq::socket_t& data_socket, zmq::socket_t& weights_socket, unsigned partId,
  unsigned layer) {
    std::cout << "FORWARD LAYER" << std::endl;
    std::vector<std::string> dataRequests{"AH" + std::to_string(layer)};
    std::vector<std::string> weightRequests{"W" + std::to_string(layer)};

    std::vector<Matrix> matrices = reqTensors(data_socket, partId, layer, dataRequests);
    std::vector<Matrix> weights = reqTensors(weights_socket, partId, layer, weightRequests);

    if (matrices.empty() || weights.empty()) {
        return constructResp(false, partId, "Got error message from server");
    }

    for (auto& M : matrices) {
        if (M.empty()){
            std::cout << M.name() << " is empty" << std::endl;
            return constructResp(false, partId, M.name() + " is empty");
        } else {
            std::cout << "GOT " << M.name() << std::endl;
        }
    }
    for (auto& W : weights) {
        if (W.empty()){
            std::cout << W.name() << " is empty" << std::endl;
            return constructResp(false, partId, W.name() + " is empty");
        } else {
            std::cout << "GOT " << W.name() << std::endl;
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

    sendTensors(data_socket, partId, layer, toSend, true);

    std::cout << "SENT tensors Z, H" << std::endl;

    // Clean up data
    for (auto& M : matrices)
        deleteMatrix(M);
    for (auto& W : weights)
        deleteMatrix(W);
    for (auto& M : toSend)
        deleteMatrix(M);

    std::cout << "Data cleaned up" << std::endl;

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

    std::cout << "Returning from function" << std::endl;

//    weights_socket.close();
//    data_socket.close();
//    ctx.close();

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
