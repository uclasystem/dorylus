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


invocation_response
apply_edge(zmq::socket_t& data_socket, zmq::socket_t& weights_socket, Chunk &chunk) {
    std::cout << "FORWARD APPLY EDGE LAYER " << chunk.layer << std::endl;

    unsigned startRequest = timestamp_ms();
    EdgeTensor eTensor = reqEdgeTensor(data_socket, chunk, "fedge");
    unsigned endRequest = timestamp_ms();
    if (eTensor.numLvids == NOT_FOUND_ERR_FIELD) {
        std::cerr << "Tensor 'fedge' was not found on graph server" << std::endl;
        return constructResp(false, chunk.localId, "Tensor 'fedge' not found");
    } else if (eTensor.numLvids == DUPLICATE_REQ_ERR_FIELD) {
        std::cerr << "Chunk already running. Request rejected" << std::endl;
        return constructResp(false, chunk.localId, "Duplicate chunk request");
    } else if (eTensor.numLvids == CHUNK_DNE_ERR) {
        std::cerr << "Chunk not found on graph server" << std::endl;
        return constructResp(false, chunk.localId, "Chunk not found");
    }

    std::cout << "GOT EDGE HEADER " << eTensor.numLvids << ", " << eTensor.numRvids
              << ", " << eTensor.featDim << std::endl;
    std::cout << "TOOK " << endRequest - startRequest << " ms" << std::endl;

//    std::vector<std::string weightRequests{"a"};
//    std::vector<Matrix> weights = reqTensors(weights_socket, chunk, weightRequests);
//    for (auto& W : weights) {
//        if (W.empty()){
//            std::cout << W.name() << " is empty" << std::endl;
//            return constructResp(false, chunk.localId, W.name() + " is empty");
//        }
//    }
//
//    if (matrices.empty() || weights.empty()) {
//        return constructResp(false, chunk.localId, "Got error message from server");
//    }

    return constructResp(false, chunk.localId, "Finished apply edge");
}

invocation_response
apply_vertex(zmq::socket_t& data_socket, zmq::socket_t& weights_socket, Chunk &chunk) {
    std::cout << "FORWARD APPLY VERTEX LAYER " << chunk.layer << std::endl;

    std::vector<std::string> dataRequests{"h"};
    std::vector<Matrix> matrices = reqTensors(data_socket, chunk, dataRequests);
    for (auto& M : matrices) {
        if (M.empty()){
            std::cout << M.name() << " is empty" << std::endl;
            return constructResp(false, chunk.localId, M.name() + " is empty");
        }
    }

    std::vector<std::string> weightRequests{"w"};
    std::cerr << "Request w" << std::endl;
    std::vector<Matrix> weights = reqTensors(weights_socket, chunk, weightRequests);
    for (auto& W : weights) {
        if (W.empty()){
            std::cout << W.name() << " is empty" << std::endl;
            return constructResp(false, chunk.localId, W.name() + " is empty");
        }
    }

    if (matrices.empty() || weights.empty()) {
        return constructResp(false, chunk.localId, "Got error message from server");
    }

    Matrix& H = matrices[0];
    Matrix& W = weights[0];

    Matrix Z = H.dot(W);
    Z.setName("z");
    deleteMatrix(H);
    deleteMatrix(W);

    std::vector<Matrix> toSend;
    toSend.push_back(Z);

    std::cout << "Sending Z tensor" << std::endl;
    int ret = sendTensors(data_socket, chunk, toSend, true);
    std::cout << "Fin send" << std::endl;

    for (auto& M : toSend)
        deleteMatrix(M);

    std::cout << "Data cleaned up" << std::endl;
    if (ret == -1) {
        return constructResp(false, chunk.localId, "This chunk is already done.");
    } else {
        return constructResp(true, chunk.localId, "Finished forward layer");
    }
    return constructResp(false, chunk.localId, "This chunk is already done.");
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
    memcpy(identity, (char *) &chunk.localId, sizeof(unsigned));
    std::srand(time(NULL));
    *(unsigned *)(identity + sizeof(unsigned)) = chunk.layer;
    *(unsigned *)(identity + sizeof(unsigned) * 2) = rand();
    memcpy(identity + sizeof(unsigned) * 3, (char *) dataserver.c_str(), dataserver.length());

    zmq::socket_t weights_socket(ctx, ZMQ_DEALER);
    zmq::socket_t data_socket(ctx, ZMQ_DEALER);
    std::cout << "Setting up comms" << std::endl;
    try {
        weights_socket.setsockopt(ZMQ_IDENTITY, identity, identity_len);
        if (RESEND) {
            weights_socket.setsockopt(ZMQ_RCVTIMEO, TIMEOUT_PERIOD);
        }
        char whost_port[50];
        sprintf(whost_port, "tcp://%s:%u", weightserver.c_str(), wport);
        weights_socket.connect(whost_port);

        data_socket.setsockopt(ZMQ_IDENTITY, identity, identity_len);
        if (RESEND) {
            data_socket.setsockopt(ZMQ_RCVTIMEO, TIMEOUT_PERIOD);
        }
        char dhost_port[50];
        sprintf(dhost_port, "tcp://%s:%u", dataserver.c_str(), dport);
        data_socket.connect(dhost_port);
    } catch(std::exception& ex) {
        return constructResp(false, chunk.localId, ex.what());
    }
    std::cout << "Finished comm setup" << std::endl;

    if (chunk.vertex != 0) {
        return apply_vertex(data_socket, weights_socket, chunk);
    } else {
        return apply_edge(data_socket, weights_socket, chunk);
    }

    std::cout << "Returning from function" << std::endl;

    weights_socket.setsockopt(ZMQ_LINGER, 0);
    weights_socket.close();
    data_socket.setsockopt(ZMQ_LINGER, 0);
    data_socket.close();
    ctx.close();

    return constructResp(false, chunk.localId, "Didn't run any config");
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
    chunk.localId = v.GetInteger("id");
    chunk.globalId = v.GetInteger("gid");
    chunk.lowBound = v.GetInteger("lb");
    chunk.upBound = v.GetInteger("ub");
    chunk.layer = v.GetInteger("layer");
    chunk.dir = static_cast<PROP_TYPE>(v.GetInteger("dir"));
    chunk.epoch = v.GetInteger("epoch");
    chunk.vertex = v.GetInteger("vtx");

    std::cout << "[ACCEPTED] Thread " << chunk.str() << " is requested from "
              << dataserver << ":" << dport << ", FORWARD layer " << chunk.layer
              << " " << chunk.vertex << std::endl;

    return apply_phase(dataserver, weightserver, dport, wport, chunk, eval);
}

int
main(int argc, char *argv[]) {
    run_handler(my_handler);

    return 0;
}
