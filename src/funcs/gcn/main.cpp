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
finalLayer(zmq::socket_t& data_socket, zmq::socket_t& weights_socket, Chunk &chunk, bool eval) {
    std::cout << "FINAL LAYER" << std::endl;
    std::vector<std::string> dataRequests{"ah", "lab"};
    std::vector<std::string> weightRequests{"w"};

    std::vector<Matrix> matrices = reqTensors(data_socket, chunk, dataRequests);
    std::vector<Matrix> weights = reqTensors(weights_socket, chunk, weightRequests);

    if (matrices.empty() || weights.empty()) {
        return constructResp(false, chunk.chunkId, "Got error message from server");
    }

    for (auto& M : matrices) {
        if (M.empty()){
            for (auto& M : matrices) deleteMatrix(M);
            for (auto& W : weights) deleteMatrix(W);

            std::cout << M.name() << " is empty" << std::endl;
            return constructResp(false, chunk.chunkId, M.name() + " is empty");
        }
    }
    for (auto& W : weights) {
        if (W.empty()){
            for (auto& M : matrices) deleteMatrix(M);
            for (auto& W : weights) deleteMatrix(W);

            std::cout << W.name() << " is empty" << std::endl;
            return constructResp(false, chunk.chunkId, W.name() + " is empty");
        }
    }

    // Forward layer
    Matrix& AH = matrices[0];
    Matrix& W = weights[0];

    Matrix Z = AH.dot(W);

    Matrix preds = softmax(Z);
    deleteMatrix(Z);
    Matrix& labels = matrices[1];

    if (eval) {
        sendAccLoss(data_socket, preds, labels, chunk);
    }

    // Backward computation
    Matrix d_out = preds - labels;
    deleteMatrix(labels);
    deleteMatrix(preds);

    Matrix interGrad = d_out.dot(W, false, true);
    deleteMatrix(W);
    Matrix d_weights = AH.dot(d_out, true, false);
    deleteMatrix(AH);
    deleteMatrix(d_out);
    interGrad.setName("grad");

    d_weights.setName("w");

    std::vector<Matrix> weightUpdates{d_weights};
    sendTensors(weights_socket, chunk, weightUpdates);

    std::vector<Matrix> toSend{interGrad};
    sendTensors(data_socket, chunk, toSend, true);

    std::cout << "SENT tensors and weight updates" << std::endl;

    // Clean up data
    for (auto& M : toSend)
        deleteMatrix(M);
    for (auto& M : weightUpdates)
        deleteMatrix(M);
    std::cout << "Data cleaned up" << std::endl;

    return constructResp(true, chunk.chunkId, "Finished final layer");
}

invocation_response
backwardLayer(zmq::socket_t& data_socket, zmq::socket_t& weights_socket, Chunk &chunk) {
    std::cout << "BACKWARD LAYER" << std::endl;
    std::vector<std::string> dataReqs{"ah", "z", "aTg"};
    std::vector<std::string> weightReqs{"w"};
    std::vector<Matrix> matrices = reqTensors(data_socket, chunk, dataReqs);
    std::vector<Matrix> weights = reqTensors(weights_socket, chunk, weightReqs);

    if (matrices.empty() || weights.empty()) {
        return constructResp(false, chunk.chunkId, "Got error message from server");
    }

    for (auto& M : matrices) {
        if (M.empty()){
            std::cout << M.name() << " is empty" << std::endl;
            return constructResp(false, chunk.chunkId, M.name() + " is empty");
        } else {
            std::cout << "GOT " << M.name() << std::endl;
        }
    }
    for (auto& W : weights) {
        if (W.empty()){
            std::cout << W.name() << " is empty" << std::endl;
            return constructResp(false, chunk.chunkId, W.name() + " is empty");
        } else {
            std::cout << "GOT " << W.name() << std::endl;
        }
    }

    Matrix& AH = matrices[0];
    Matrix& Z = matrices[1];
    Matrix& grad = matrices[2];

    Matrix& W = weights[0];

    Matrix actDeriv = tanhDerivative(Z);
    deleteMatrix(Z);

    Matrix interGrad = grad * actDeriv;
    deleteMatrix(grad);
    deleteMatrix(actDeriv);

    Matrix resultGrad = interGrad.dot(W, false, true);
    deleteMatrix(W);

    Matrix d_weights = AH.dot(interGrad, true, false);
    deleteMatrix(AH);
    deleteMatrix(interGrad);

    d_weights.setName("w");
    resultGrad.setName("grad");

    std::vector<Matrix> weightUpdates{d_weights};
    sendTensors(weights_socket, chunk, weightUpdates);

    std::vector<Matrix> toSend{resultGrad};
    if (chunk.layer != 0) {
        sendTensors(data_socket, chunk, toSend, true);
    } else { // the last backward layer (layer 0), skip sending the grad back
        sendFinMsg(data_socket, chunk);
    }

    for (auto& M : toSend)
        deleteMatrix(M);
    for (auto& M : weightUpdates)
        deleteMatrix(M);

    return constructResp(true, chunk.chunkId, "Finished backward layer");
}

invocation_response
forwardLayer(zmq::socket_t& data_socket, zmq::socket_t& weights_socket, Chunk &chunk) {
    std::cout << "FORWARD LAYER" << std::endl;
    std::vector<std::string> dataRequests{"ah"};
    std::vector<std::string> weightRequests{"w"};

    std::cerr << "req data" << std::endl;
    std::vector<Matrix> matrices = reqTensors(data_socket, chunk, dataRequests);
    std::cerr << "fin data\nreq weights" << std::endl;
    std::vector<Matrix> weights = reqTensors(weights_socket, chunk, weightRequests);
    std::cerr << "fin weights" << std::endl;

    if (matrices.empty() || weights.empty()) {
        return constructResp(false, chunk.chunkId, "Got error message from server");
    }

    for (auto& M : matrices) {
        if (M.empty()){
            std::cout << M.name() << " is empty" << std::endl;
            return constructResp(false, chunk.chunkId, M.name() + " is empty");
        }
    }
    for (auto& W : weights) {
        if (W.empty()){
            std::cout << W.name() << " is empty" << std::endl;
            return constructResp(false, chunk.chunkId, W.name() + " is empty");
        }
    }

    Matrix& AH = matrices[0];
    Matrix& W = weights[0];

    Matrix Z = AH.dot(W);
    Z.setName("z");
    deleteMatrix(AH);
    deleteMatrix(W);

    Matrix H_l = tanh(Z);
    H_l.setName("h");

    std::vector<Matrix> toSend;
    toSend.push_back(Z);
    toSend.push_back(H_l);

    sendTensors(data_socket, chunk, toSend, true);

    std::cout << "SENT tensors Z, H" << std::endl;

    // Clean up data
    for (auto& M : toSend)
        deleteMatrix(M);
    std::cout << "Data cleaned up" << std::endl;

    return constructResp(true, chunk.chunkId, "Finished forward layer");
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
apply_phase(std::string dataserver, std::string weightserver, unsigned dport, unsigned wport, Chunk &chunk, bool eval) {
    zmq::context_t ctx(2);

    // Creating identity
    size_t identity_len = sizeof(unsigned) * 3 + dataserver.length();
    char identity[identity_len];
    memcpy(identity, (char *) &chunk.chunkId, sizeof(unsigned));
    std::srand(time(NULL));
    *(unsigned *)(identity + sizeof(unsigned)) = chunk.layer;
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
        return constructResp(false, chunk.chunkId, ex.what());
    }

    if (chunk.dir == PROP_TYPE::FORWARD && chunk.layer < 1) {
        return forwardLayer(data_socket, weights_socket, chunk);
    } else if (chunk.dir == PROP_TYPE::FORWARD && chunk.layer == 1) {
        return finalLayer(data_socket, weights_socket, chunk, eval);
    } else if (chunk.dir == PROP_TYPE::BACKWARD) {
        return backwardLayer(data_socket, weights_socket, chunk);
    }

    std::cout << "Returning from function" << std::endl;

//    weights_socket.close();
//    data_socket.close();
//    ctx.close();

    return constructResp(false, chunk.chunkId, "Didn't run any config");
}


/** Handler that hooks with lambda API. */
invocation_response
my_handler(invocation_request const& request) {
    JsonValue json(request.payload);
    auto v = json.View();

    std::string dataserver = v.GetString("dserver");
    std::string weightserver = v.GetString("wserver");
    unsigned dport = v.GetInteger("dport");
    unsigned wport = v.GetInteger("wport");
    bool eval = v.GetBool("eval");

    Chunk chunk;
    chunk.chunkId = v.GetInteger("id");
    chunk.lowBound = v.GetInteger("lb");
    chunk.upBound = v.GetInteger("ub");
    chunk.layer = v.GetInteger("layer");
    chunk.dir = static_cast<PROP_TYPE>(v.GetInteger("dir"));
    chunk.epoch = v.GetInteger("epoch");
    chunk.vertex = v.GetInteger("vtx");

    std::cout << "[ACCEPTED] Thread " << chunk.chunkId << " is requested from "
              << dataserver << ":" << dport << ", FORWARD layer " << chunk.layer
              << "." << std::endl;

    return apply_phase(dataserver, weightserver, dport, wport, chunk, eval);
}

int
main(int argc, char *argv[]) {
    run_handler(my_handler);

    return 0;
}
