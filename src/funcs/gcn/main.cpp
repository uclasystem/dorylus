#include <algorithm>
#include <chrono>
#include <ratio>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <cmath>
#include <unistd.h>

#include <cblas.h>
#include <zmq.hpp>

#include <aws/lambda-runtime/runtime.h>
#include <aws/core/Aws.h>
#include <aws/core/utils/json/JsonSerializer.h>

#include "../../common/matrix.hpp"
#include "../../common/utils.hpp"

#include "forward_ops.hpp"
#include "backward_ops.hpp"
#include "network_ops.hpp"


#define SLEEP_PERIOD   1000  // us
#define TIMEOUT_PERIOD (500) // ms

#define SND_MORE true
#define NO_MORE false

using namespace Aws::Utils::Json;
using namespace aws::lambda_runtime;
using namespace std::chrono;

bool lastLayer = false;

static Matrix
forwardLayer(zmq::socket_t& data_socket, zmq::socket_t& weight_socket,
  unsigned partId, unsigned layer) {
    try {
        Matrix weights, feats, z, activations;

        // Request weights matrix of the current layer.
        std::thread t([&] {     // Weight requests run in a separate thread.
            std::cout << "< FORWARD > Asking weightserver..." << whost_port << std::endl;
            do {
                weights = requestMatrix(weights_socket, OP::PULL_FORWARD, layer);
            } while (weights.empty());
            std::cout << "< FORWARD > Got data from weightserver." << dhost_port << std::endl;
        });

        // Request feature activation matrix of the current layer.
        std::cout << "< FORWARD > Asking dataserver..." << std::endl;
        reqStart = timestamp_ms();
        do {
            feats = requestMatrix(data_socket, OP::PULL_FORWARD, id, true);
        } while (feats.empty());
        reqEnd = timestamp_ms();
        std::cout << "< FORWARD > Got data from dataserver." << std::endl;

        t.join();

        if (weights.empty()) {
            JsonValue jsonResponse;
            jsonResponse.WithBool("success", false);
            jsonResponse.WithString("reason", "Weights empty");

            auto response = jsonResponse.View().WriteCompact();
            return invocation_response::failure(response, "application/json");
        }
        if (feats.empty()) {
            JsonValue jsonResponse;
            jsonResponse.WithBool("success", false);
            jsonResponse.WithString("reason", "Features empty");

            auto response = jsonResponse.View().WriteCompact();
            return invocation_response::failure(response, "appliation/json");
        }

        // Multiplication.
        z = feats.dot(weights);
        activations = softmax(z);

        sendMatrices(z, activations, data_socket, id);

        // Delete malloced spaces.
        delete[] weights.getData();
        delete[] feats.getData();
        delete[] z.getData();
        delete[] activations.getData();
    } catch(std::exception &ex) {
        JsonValue jsonResponse;
        jsonResponse.WithBool("success", false);
        jsonResponse.WithString("reason", ex.what());

        auto response = jsonResponse.View().WriteCompact();
        return invocation_response::failure(response, "application/json");
    }
}

static Matrix
finalLayer(zmq::socket_t& data_socket, zmq::socket_t& weight_socket,
  unsigned partId, unsigned layer) {
    try {
        Matrix weights, feats, z, activations;

        // Request weights matrix of the current layer.
        std::thread t([&] {     // Weight requests run in a separate thread.
            std::cout << "< FORWARD > Asking weightserver..." << whost_port << std::endl;
            do {
                weights = requestMatrix(weights_socket, OP::PULL_FORWARD, layer);
            } while (weights.empty());
            std::cout << "< FORWARD > Got data from weightserver." << dhost_port << std::endl;
        });

        // Request feature activation matrix of the current layer.
        std::cout << "< FORWARD > Asking dataserver..." << std::endl;
        reqStart = timestamp_ms();
        do {
            feats = requestMatrix(data_socket, OP::PULL_FORWARD, id, true);
        } while (feats.empty());
        reqEnd = timestamp_ms();
        std::cout << "< FORWARD > Got data from dataserver." << std::endl;

        t.join();

        if (weights.empty()) {
            JsonValue jsonResponse;
            jsonResponse.WithBool("success", false);
            jsonResponse.WithString("reason", "Weights empty");

            auto response = jsonResponse.View().WriteCompact();
            return invocation_response::failure(response, "application/json");
        }
        if (feats.empty()) {
            JsonValue jsonResponse;
            jsonResponse.WithBool("success", false);
            jsonResponse.WithString("reason", "Features empty");

            auto response = jsonResponse.View().WriteCompact();
            return invocation_response::failure(response, "appliation/json");
        }

        // Multiplication.
        z = feats.dot(weights);
        activations = softmax(z);

        sendMatrices(z, activations, data_socket, id);

        // Delete malloced spaces.
        delete[] weights.getData();
        delete[] feats.getData();
        delete[] z.getData();
        delete[] activations.getData();
    } catch(std::exception &ex) {
        JsonValue jsonResponse;
        jsonResponse.WithBool("success", false);
        jsonResponse.WithString("reason", ex.what());

        auto response = jsonResponse.View().WriteCompact();
        return invocation_response::failure(response, "application/json");
    }
}

static Matrix
backwardLayer(zmq::socket_t& data_socket, zmq::socket_t& weight_socket,
  unsigned partId, unsigned layer) {
    // gradLayer
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
    } catch(std::exception& ex) {
        JsonValue jsonResponse;
        jsonResponse.WithBool("success", false);
        jsonResponse.WithString("reason", ex.what());

        auto response = jsonResponse.View().WriteCompact();
        return invocation_response::failure(response, "application/json");
    }

    if (prop_dir == PROP_TYPE::FORWARD && !lastLayer) {
        forwardLayer(data_socket, weight_socket, id, layer);
    } else if (prop_dir == PROP_TYPE::FORWARD && lastLayer) {
        finalLayer(data_socket, weight_socket, id, layer);
    } else if (prop_dir == PROP_TYPE::BACKWARD) {
        backwardLayer(data_socket, weight_socket, id, layer);
    }

    try {
        Matrix weights, feats, z, activations;

        // Request weights matrix of the current layer.
        std::thread t([&] {     // Weight requests run in a separate thread.
            std::cout << "< FORWARD > Asking weightserver..." << whost_port << std::endl;
            do {
                weights = requestMatrix(weights_socket, OP::PULL_FORWARD, layer);
            } while (weights.empty());
            std::cout << "< FORWARD > Got data from weightserver." << dhost_port << std::endl;
        });

        // Request feature activation matrix of the current layer.
        std::cout << "< FORWARD > Asking dataserver..." << std::endl;
        reqStart = timestamp_ms();
        do {
            feats = requestMatrix(data_socket, OP::PULL_FORWARD, id, true);
        } while (feats.empty());
        reqEnd = timestamp_ms();
        std::cout << "< FORWARD > Got data from dataserver." << std::endl;

        t.join();

        if (weights.empty()) {
            JsonValue jsonResponse;
            jsonResponse.WithBool("success", false);
            jsonResponse.WithString("reason", "Weights empty");

            auto response = jsonResponse.View().WriteCompact();
            return invocation_response::failure(response, "application/json");
        }
        if (feats.empty()) {
            JsonValue jsonResponse;
            jsonResponse.WithBool("success", false);
            jsonResponse.WithString("reason", "Features empty");

            auto response = jsonResponse.View().WriteCompact();
            return invocation_response::failure(response, "appliation/json");
        }

        // Multiplication.
        z = feats.dot(weights);


        if (lastLayer) {
            activations = softmax(z);
        } else {
            activations = activate(z);
        }

        sendMatrices(z, activations, data_socket, id);

        // Delete malloced spaces.
        delete[] weights.getData();
        delete[] feats.getData();
        delete[] z.getData();
        delete[] activations.getData();
    } catch(std::exception &ex) {
        JsonValue jsonResponse;
        jsonResponse.WithBool("success", false);
        jsonResponse.WithString("reason", ex.what());

        auto response = jsonResponse.View().WriteCompact();
        return invocation_response::failure(response, "application/json");
    }

    JsonValue jsonResponse;
    jsonResponse.WithBool("success", true);

    auto response = jsonResponse.View().WriteCompact();
    return invocation_response::success(response, "application/json");
}


/** Handler that hooks with lambda API. */
static invocation_response
my_handler(invocation_request const& request) {
    lambdaStart = timestamp_ms();
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
