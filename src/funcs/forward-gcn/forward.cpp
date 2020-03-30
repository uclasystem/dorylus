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
#include "../../../src/common/matrix.hpp"
#include "../../../src/common/utils.hpp"

#define SLEEP_PERIOD   1000  // us
#define TIMEOUT_PERIOD (500) // ms

#define SND_MORE true
#define NO_MORE false


using namespace aws::lambda_runtime;
using namespace Aws::Utils::Json;

// timestamp for profiling
unsigned timestamps[30];
unsigned tsidx;

static invocation_response
constructResp(bool success, unsigned id, std::string msg) {
    JsonValue jsonResponse;
    jsonResponse.WithBool("success", success);
    jsonResponse.WithInteger("type", PROP_TYPE::FORWARD);
    jsonResponse.WithInteger("id", id);
    jsonResponse.WithString("message", msg);
    auto response = jsonResponse.View().WriteCompact();
    return invocation_response::success(response, "appliation/json");
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
static invocation_response
forward_prop_layer(std::string dataserver, std::string weightserver, unsigned dport, unsigned wport,
                   unsigned id, unsigned layer, bool lastLayer) {
    zmq::context_t ctx(1);
    std::srand(time(NULL));

    //
    // Lambda socket identity is set to:
    //
    //      [ 4 Bytes partId ] | [ 4 Bytes layer ] | [ 4 Bytes rand ] | [ n Bytes the string of dataserverIp ]
    //
    // One should extract the partition id by reading the first 4 Bytes, which is simply parse<unsigned>(...).
    //
    size_t identity_len = sizeof(unsigned) * 3 + dataserver.length();
    char identity[identity_len];
    *(unsigned *)identity = id;
    *(unsigned *)(identity + sizeof(unsigned)) = layer;
    *(unsigned *)(identity + sizeof(unsigned) * 2) = rand();
    memcpy(identity + sizeof(unsigned) * 3, (char *) dataserver.c_str(), dataserver.length());

    Matrix weights, feats, z, activations;
    try {
        zmq::socket_t weights_socket(ctx, ZMQ_DEALER);
        weights_socket.setsockopt(ZMQ_IDENTITY, identity, identity_len);
        weights_socket.setsockopt(ZMQ_TCP_KEEPALIVE, 1);
        char whost_port[50];
        sprintf(whost_port, "tcp://%s:%u", weightserver.c_str(), wport);
        weights_socket.connect(whost_port);

        zmq::socket_t data_socket(ctx, ZMQ_DEALER);
        data_socket.setsockopt(ZMQ_IDENTITY, identity, identity_len);
        data_socket.setsockopt(ZMQ_TCP_KEEPALIVE, 1);
        char dhost_port[50];
        sprintf(dhost_port, "tcp://%s:%u", dataserver.c_str(), dport);
        data_socket.connect(dhost_port);

        std::thread wReqThd([&] {
            __sync_synchronize();
            // Request weights matrix of the current layer.
            std::cout << "< FORWARD > Asking weightserver..." << whost_port << std::endl;
            weights = requestWeight(weights_socket, OP::PULL_VTX_FORWARD, layer);
            std::cout << "< FORWARD > Got data from weightserver." << dhost_port << std::endl;
        });

        // Request feature activation matrix of the current layer.
        std::cout << "< FORWARD > Asking dataserver..." << std::endl;
        set_timestamp();
        feats = requestMatrix(data_socket, OP::PULL_VTX_FORWARD, id);
        set_timestamp();
        std::cout << "< FORWARD > Got data from dataserver." << std::endl;

        wReqThd.join();

        if (weights.empty() || feats.empty()) {
            auto resp = constructResp(false, id, std::string("Weights ") + (weights.empty() ? "are" : "are not") +
                                    " empty, Feats " + (feats.empty() ? "are" : "are not") +
                                    " empty. Stopped by graph server");
            deleteMatrix(weights);
            deleteMatrix(feats);
            return resp;
        }

        // Multiplication.
        std::cout << "< FORWARD > Begin computation." << std::endl;
        z = feats.dot(weights);

        if (lastLayer) {
            activations = softmax(z);
        } else {
            activations = activate(z);
        }
        std::cout << "< FORWARD > Finish computation." << std::endl;

        std::cout << "< FORWARD > Send results back." << std::endl;
        set_timestamp();
        sendMatrices(z, activations, data_socket, id);
        set_timestamp();
        std::cout << "< FORWARD > Send results finished." << std::endl;

        // Delete malloced spaces.
        deleteMatrix(weights);
        deleteMatrix(feats);
        deleteMatrix(z);
        deleteMatrix(activations);
    } catch(std::exception &ex) {
        deleteMatrix(weights);
        deleteMatrix(feats);
        deleteMatrix(z);
        deleteMatrix(activations);
        auto resp = constructResp(false, id, ex.what());
        return resp;
    }

    Aws::String tsStr = "";
    for (unsigned i = 0; i < tsidx; ++i) {
        tsStr += std::to_string(timestamps[i]) + " ";
    }

    auto resp = constructResp(true, id, tsStr);
    return resp;
}


/** Handler that hooks with lambda API. */
static invocation_response
my_handler(invocation_request const& request) {
    tsidx = 0;

    JsonValue json(request.payload);
    auto v = json.View();

    std::string dataserver = v.GetString("dataserver");
    std::string weightserver = v.GetString("weightserver");
    unsigned dport = v.GetInteger("dport");
    unsigned wport = v.GetInteger("wport");
    unsigned layer = v.GetInteger("layer");
    unsigned chunkId = v.GetInteger("id");
    bool lastLayer = v.GetBool("lastLayer");

    std::cout << "[ACCEPTED] Thread " << chunkId << " is requested from " << dataserver << ":" << dport
              << ", FORWARD layer " << layer << "." << std::endl;

    return forward_prop_layer(dataserver, weightserver, dport, wport, chunkId, layer, lastLayer);
}

int
main(int argc, char *argv[]) {
    run_handler(my_handler);

    return 0;
}
