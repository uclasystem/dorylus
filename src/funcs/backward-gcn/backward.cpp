#include <chrono>
#include <ratio>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <algorithm>
#include <cmath>
#include <cblas.h>
#include <zmq.hpp>
#include <aws/lambda-runtime/runtime.h>
#include <aws/core/Aws.h>
#include <aws/core/utils/json/JsonSerializer.h>

#include <time.h>
#include <iomanip>

#include "../../common/matrix.hpp"
#include "../../common/utils.hpp"

#define SLEEP_PERIOD 1000
#define TIMEOUT_PERIOD 500

#define SND_MORE true
#define NO_MORE false

// timestamp for profiling
unsigned timestamps[30];
unsigned tsidx = 0;

void set_timestamp() {
    using namespace std::chrono;
    auto now = system_clock::now().time_since_epoch();
    unsigned me = duration_cast<milliseconds>(now).count() - BASE_TMSP;
    timestamps[tsidx++] = me;
}

using namespace Aws::Utils::Json;
using namespace aws::lambda_runtime;

bool lastLayer = false;

int
requestTensors(zmq::socket_t& socket, OP op, unsigned partId, unsigned layer, std::vector<Matrix> &mats) {
    zmq::message_t header(HEADER_SIZE);
    populateHeader((char*) header.data(), op, partId, layer);
    socket.send(header);

    // Timer reqTimer;
    // reqTimer.start();

    zmq::message_t respHeader;
    socket.recv(&respHeader);
    // while(!socket.recv(&respHeader, ZMQ_DONTWAIT)) {
    //     usleep(SLEEP_PERIOD);
    //     if (reqTimer.peek() > TIMEOUT_PERIOD) {
    //         // zmq::message_t _hdr(HEADER_SIZE);
    //         // populateHeader((char *) _hdr.data(), op, partId, type, layer);
    //         zmq::message_t _hdr;
    //         _hdr.copy(&header);
    //         socket.send(_hdr);
    //         reqTimer.start();
    //     }
    // }

    unsigned layerResp = parse<unsigned>((char*) respHeader.data(), 1);
    if (layerResp != 0) {
        std::cerr << "[ ERROR ] Discard execution." << std::endl;
        std::cerr << "[ ERROR ] No corresponding matrix" << std::endl;
        return -1;
    } else {
        unsigned rows = parse<unsigned>((char*) respHeader.data(), 2);
        unsigned cols;

        {
            unsigned recv_ts = *((unsigned *)respHeader.data() + 5);
            timestamps[tsidx++] = recv_ts;
            unsigned send_ts = *((unsigned *)respHeader.data() + 6);
            timestamps[tsidx++] = send_ts;
        }

        zmq::message_t matxData;
        for (unsigned i = 0; i < 3; ++i) {
            socket.recv(&matxData);
            cols = matxData.size() / sizeof(FeatType) / rows;
            printf("recved %u matrix: (%u, %u)\n", i, rows, cols);
            FeatType *matxBuffer = new FeatType[rows * cols];
            std::memcpy(matxBuffer, matxData.data(), matxData.size());
            mats.push_back(Matrix(rows, cols, matxBuffer));
        }

        return 0;
    }
}


static Matrix
requestWeight(zmq::socket_t& socket, OP op, unsigned layer) {
    zmq::message_t header(HEADER_SIZE);
    populateHeader((char*) header.data(), op, layer);
    socket.send(header);

    // Timer reqTimer;
    // reqTimer.start();

    zmq::message_t respHeader;
    socket.recv(&respHeader);
    // while(!socket.recv(&respHeader, ZMQ_DONTWAIT)) {
    //     usleep(SLEEP_PERIOD);
    //     if (reqTimer.peek() > TIMEOUT_PERIOD) {
    //         // zmq::message_t _hdr(HEADER_SIZE);
    //         // populateHeader((char *) _hdr.data(), op, partId, type, layer);
    //         zmq::message_t _hdr;
    //         _hdr.copy(&header);
    //         socket.send(_hdr);
    //         reqTimer.start();
    //     }
    // }

    unsigned layerResp = parse<unsigned>((char*) respHeader.data(), 1);
    if (layerResp != 0) {
        std::cerr << "[ ERROR ] No corresponding matrix" << std::endl;
        return Matrix();
    } else {
        unsigned rows = parse<unsigned>((char*) respHeader.data(), 2);
        unsigned cols = parse<unsigned>((char*) respHeader.data(), 3);

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
sendMatrix(Matrix& matrix, zmq::socket_t& socket, unsigned id, bool data = false) {
    // Send push header.
    zmq::message_t header(HEADER_SIZE);
    populateHeader((char *) header.data(), OP::PUSH_BACKWARD, id, matrix.getRows(),
                    matrix.getCols());
    socket.send(header, ZMQ_SNDMORE);

    zmq::message_t updateMsg(matrix.getDataSize());
    std::memcpy(updateMsg.data(), matrix.getData(), matrix.getDataSize());
    socket.send(updateMsg);

    // Timer sndTimer;
    // sndTimer.start();

    // Wait for updates settled reply.
    zmq::message_t confirm;
    socket.recv(&confirm);
    // while(!socket.recv(&confirm, ZMQ_DONTWAIT)) {
    //     usleep(SLEEP_PERIOD);
    //     if (sndTimer.peek() > TIMEOUT_PERIOD) {
    //         // zmq::message_t _hdr(HEADER_SIZE);
    //         // populateHeader((char *) _hdr.data(), OP::PUSH_BACKWARD, id, matrix.getRows(), matrix.getCols());
    //         zmq::message_t _hdr;
    //         _hdr.copy(&header);
    //         socket.send(_hdr, ZMQ_SNDMORE);
    //         // zmq::message_t _updMsg(matrix.getDataSize());
    //         // std::memcpy(_updMsg.data(), matrix.getData(), matrix.getDataSize());
    //         zmq::message_t _updMsg;
    //         _updMsg.copy(&updateMsg);
    //         socket.send(_updMsg);
    //         sndTimer.start();
    //     }
    // }

    if (data) {
        unsigned recv_ts = *((unsigned *)confirm.data());
        timestamps[tsidx++] = recv_ts;
        unsigned send_ts = *((unsigned *)confirm.data() + 1);
        timestamps[tsidx++] = send_ts;
    }
}

static void
deleteMatrix(Matrix &mat) {
    if (!mat.empty()) {
        delete[] mat.getData();
        mat = Matrix();
    }
}


/**
 *
 * Apply derivative of the activation function on a matrix.
 *
 */
static Matrix
activateDerivative(Matrix& mat) {
    FeatType *res = new FeatType[mat.getNumElemts()];
    FeatType *zData = mat.getData();

    for (unsigned i = 0; i < mat.getNumElemts(); ++i)
        res[i] = 1 - std::pow(std::tanh(zData[i]), 2);

    return Matrix(mat.getRows(), mat.getCols(), res);
}

static invocation_response
gradLoss(zmq::socket_t& data_socket, zmq::socket_t& weight_socket, unsigned id, unsigned layer) {
    Matrix weights;
    std::vector<Matrix> savedTensors;

    try {
        std::cout << "< BACKWARD > Requesting weights" << std::endl;
        weights = requestWeight(weight_socket, OP::PULL_BACKWARD, layer);

        std::cout << "< BACKWARD > Getting savedTensors" << std::endl;
        int ret = 0;
        set_timestamp();
        ret = requestTensors(data_socket, OP::PULL_BACKWARD, id, layer, savedTensors);
        set_timestamp();

        if (ret != 0 || weights.empty()) {
            JsonValue jsonResponse;
            jsonResponse.WithBool("success", false);
            jsonResponse.WithInteger("type", PROP_TYPE::BACKWARD);
            jsonResponse.WithInteger("id", id);
            jsonResponse.WithString("reason", std::string("Weights ") + (weights.empty() ? "are" : "are not") +
                                    " empty, Feats " + (ret != 0 ? "are" : "are not") +
                                    " empty. Stopped by graph server");

            deleteMatrix(weights);
            for (Matrix &mat : savedTensors) {
                deleteMatrix(mat);
            }
            auto response = jsonResponse.View().WriteCompact();
            return invocation_response::success(response, "appliation/json");
        }

        Matrix &predictions = savedTensors[0];
        Matrix &labels = savedTensors[1];
        Matrix &ah = savedTensors[2];

        // derivative of softmax
        std::cout << "< BACKWARD > Calculating cross entropy" << std::endl;
        Matrix d_output = predictions - labels;
        deleteMatrix(predictions);
        deleteMatrix(labels);

        // d_out * W^T
        std::cout << "< BACKWARD > Computing gradient" << std::endl;
        Matrix interGrad = d_output.dot(weights, false, true);
        deleteMatrix(weights);

        // AH^T * d_out
        std::cout << "< BACKWARD > Computing weight updates" << std::endl;
        Matrix weightUpdates = ah.dot(d_output, true, false);
        deleteMatrix(ah);
        deleteMatrix(d_output);

        std::cout << "< BACKWARD > Sending gradient to graph server" << std::endl;
        set_timestamp();
        sendMatrix(interGrad, data_socket, id, true);
        set_timestamp();
        deleteMatrix(interGrad);

        std::cout << "< BACKWARD > Sending weight updates" << std::endl;
        sendMatrix(weightUpdates, weight_socket, layer);
        deleteMatrix(weightUpdates);
    } catch(std::exception &ex) {
        JsonValue jsonResponse;
        jsonResponse.WithBool("success", false);
        jsonResponse.WithInteger("type", PROP_TYPE::BACKWARD);
        jsonResponse.WithInteger("id", id);
        jsonResponse.WithString("reason", ex.what());

        deleteMatrix(weights);
        for (Matrix &mat : savedTensors) {
            deleteMatrix(mat);
        }
        auto response = jsonResponse.View().WriteCompact();
        return invocation_response::success(response, "application/json");
    }

    Aws::String tsStr = "";
    for (unsigned i = 0; i < tsidx; i++) {
        tsStr += std::to_string(timestamps[i]) + " ";
    }
    JsonValue jsonResponse;
    jsonResponse.WithBool("success", true);
    jsonResponse.WithInteger("type", PROP_TYPE::BACKWARD);
    jsonResponse.WithInteger("id", id);
    jsonResponse.WithString("timestamp", tsStr);
    auto response = jsonResponse.View().WriteCompact();
    return invocation_response::success(response, "application/json");
}

static invocation_response
gradLayer(zmq::socket_t& data_socket, zmq::socket_t& weight_socket, unsigned id, unsigned layer) {
    Matrix weights;
    std::vector<Matrix> savedTensors;

    try {
        // REQUESTING ALL NEEDED TENSORS FOR COMPUTATION
        std::cout << "< BACKWARD > Requesting weights" << std::endl;
        weights = requestWeight(weight_socket, OP::PULL_BACKWARD, layer);

        std::cout << "< BACKWARD > Requesting savedTensors" << std::endl;
        set_timestamp();
        int ret = 0;
        ret = requestTensors(data_socket, OP::PULL_BACKWARD, id, layer, savedTensors);
        set_timestamp();

        if (ret != 0 || weights.empty()) {
            JsonValue jsonResponse;
            jsonResponse.WithBool("success", false);
            jsonResponse.WithInteger("type", PROP_TYPE::BACKWARD);
            jsonResponse.WithInteger("id", id);
            jsonResponse.WithString("reason", std::string("Weights ") + (weights.empty() ? "are" : "are not") +
                                    " empty, Feats " + (ret != 0 ? "are" : "are not") +
                                    " empty. Stopped by graph server");

            deleteMatrix(weights);
            for (Matrix &mat : savedTensors) {
                deleteMatrix(mat);
            }
            auto response = jsonResponse.View().WriteCompact();
            return invocation_response::success(response, "appliation/json");
        }

        Matrix &grad = savedTensors[0];
        Matrix &z = savedTensors[1];
        Matrix &ah = savedTensors[2];
        // END REQUESTING ALL NEEDED TENSORS FOR COMPUTATION

        // BACKWARDS COMPUTATION
        std::cout << "< BACKWARD > Calculating derivative of activation "
                << z.shape() << std::endl;
        Matrix actDeriv = activateDerivative(z);
        deleteMatrix(z);

        std::cout << "< BACKWARD > Hadamard multiplication" << grad.shape() << " "
                << actDeriv.shape() << std::endl;
        Matrix interGrad = grad * actDeriv;
        deleteMatrix(grad);
        deleteMatrix(actDeriv);

        std::cout << "< BACKWARD > MatMul(gradient, weights) " << interGrad.shape() << " "
                << weights.shape() << std::endl;
        Matrix resultGrad = interGrad.dot(weights, false, true);
        deleteMatrix(weights);

        std::cout << "< BACKWARD > Computing weight updates " << ah.shape() << " "
                << interGrad.shape() << std::endl;
        Matrix weightUpdates = ah.dot(interGrad, true, false);
        deleteMatrix(ah);
        deleteMatrix(interGrad);
        // END BACKWARDS COMPUTATION
        // SENDING BACKWARDS RESULTS

        std::cout << "< BACKWARD > Sending gradient to graph server" << std::endl;
        set_timestamp();
        sendMatrix(resultGrad, data_socket, id, true);
        set_timestamp();
        deleteMatrix(resultGrad);

        std::cout << "< BACKWARD > Sending weight updates" << std::endl;
        sendMatrix(weightUpdates, weight_socket, layer);
        deleteMatrix(weightUpdates);
        // END SENDING BACKWARDS RESULTS
    } catch(std::exception &ex) {
        JsonValue jsonResponse;
        jsonResponse.WithBool("success", false);
        jsonResponse.WithInteger("type", PROP_TYPE::BACKWARD);
        jsonResponse.WithInteger("id", id);
        jsonResponse.WithString("reason", ex.what());

        deleteMatrix(weights);
        for (Matrix &mat : savedTensors) {
            deleteMatrix(mat);
        }
        auto response = jsonResponse.View().WriteCompact();
        return invocation_response::success(response, "application/json");
    }

    Aws::String tsStr = "";
    for (unsigned i = 0; i < tsidx; i++) {
        tsStr += std::to_string(timestamps[i]) + " ";
    }
    JsonValue jsonResponse;
    jsonResponse.WithBool("success", true);
    jsonResponse.WithInteger("type", PROP_TYPE::BACKWARD);
    jsonResponse.WithInteger("id", id);
    jsonResponse.WithString("timestamp", tsStr);
    auto response = jsonResponse.View().WriteCompact();
    return invocation_response::success(response, "application/json");
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
backward_prop(std::string dataserver, std::string weightserver, unsigned dport,
              unsigned wport, unsigned id, unsigned layer, bool lastLayer) {
    zmq::context_t ctx(1);

    //
    // Lambda socket identity is set to:
    //
    //      [ 4 Bytes partId ] | [ n Bytes the string of dataserverIp ]
    //
    // One should extract the partition id by reading the first 4 Bytes, which is simply parse<unsigned>(...).
    //
    size_t identity_len = sizeof(unsigned) * 3 + dataserver.length();
    char identity[identity_len];
    memcpy(identity, (char *) &id, sizeof(unsigned));
    std::srand(time(NULL));
    *(unsigned *)(identity + sizeof(unsigned)) = layer;
    *(unsigned *)(identity + sizeof(unsigned) * 2) = rand();
    memcpy(identity + sizeof(unsigned) * 3, (char *) dataserver.c_str(), dataserver.length());

    zmq::socket_t weight_socket(ctx, ZMQ_DEALER);
    weight_socket.setsockopt(ZMQ_IDENTITY, identity, identity_len);
    char whost_port[50];
    sprintf(whost_port, "tcp://%s:%u", weightserver.c_str(), wport);
    weight_socket.connect(whost_port);

    zmq::socket_t data_socket(ctx, ZMQ_DEALER);
    data_socket.setsockopt(ZMQ_IDENTITY, identity, identity_len);
    char dhost_port[50];
    sprintf(dhost_port, "tcp://%s:%u", dataserver.c_str(), dport);
    data_socket.connect(dhost_port);
    if (lastLayer) {
        std::cout << "< BACKWARD > Computing gradient from loss" << std::endl;
        return gradLoss(data_socket, weight_socket, id, layer);
    } else {
        std::cout << "< BACKWARD > Computing gradient for this layer" << std::endl;
        return gradLayer(data_socket, weight_socket, id, layer);
    }

    // impossible to reach here
    JsonValue jsonResponse;
    jsonResponse.WithBool("success", true);
    jsonResponse.WithInteger("type", PROP_TYPE::BACKWARD);
    jsonResponse.WithInteger("id", id);
    auto response = jsonResponse.View().WriteCompact();
    return invocation_response::success(response, "application/json");
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
    lastLayer = v.GetBool("lastLayer");

    std::cout << "[ACCEPTED] Thread " << chunkId << " is requested from " <<
        dataserver << ":" << dport << ", BACKWARD on layer " << layer
        << std::endl;

    return backward_prop(dataserver, weightserver, dport, wport, chunkId, layer, lastLayer);
}

int
main(int argc, char *argv[]) {
    run_handler(my_handler);

    return 0;
}
