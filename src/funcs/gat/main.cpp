#include <algorithm>
#include <cassert>
#include <chrono>
#include <ratio>
#include <iostream>
#include <memory>
#include <sstream>
#include <random>
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


#define IDENTITY_SIZE (sizeof(Chunk) + sizeof(unsigned))
std::vector<char> constructIdentity(Chunk &chunk) {
    std::vector<char> identity(IDENTITY_SIZE);

    std::random_device rd;
    std::mt19937 generator(rd());

    unsigned rand = generator();
    std::cout << "RAND " << rand << std::endl;
    std::memcpy(identity.data(), &chunk, sizeof(chunk));
    std::memcpy(identity.data() + sizeof(chunk), &rand, sizeof(unsigned));

    return identity;
}

invocation_response
apply_edge(zmq::socket_t& data_socket, zmq::socket_t& weights_socket, Chunk &chunk) {
    std::cout << "FORWARD APPLY EDGE LAYER " << chunk.layer << std::endl;

    unsigned startRequest = timestamp_ms();
    EdgeInfo eInfo = reqEdgeInfo(data_socket, chunk);
    unsigned endRequest = timestamp_ms();
    if (eInfo.numLvids == NOT_FOUND_ERR_FIELD) {
        std::cerr << "Tensor 'fedge' was not found on graph server" << std::endl;
        return constructResp(false, chunk.localId, "Tensor 'fedge' not found");
    } else if (eInfo.numLvids == DUPLICATE_REQ_ERR_FIELD) {
        std::cerr << "Chunk already running. Request rejected" << std::endl;
        return constructResp(false, chunk.localId, "Duplicate chunk request");
    } else if (eInfo.numLvids == CHUNK_DNE_ERR) {
        std::cerr << "Chunk not found on graph server" << std::endl;
        return constructResp(false, chunk.localId, "Chunk not found");
    } else if (eInfo.numLvids == ERR_HEADER_FIELD) {
        std::cerr << "Prorably a null pointer? I dunno" << std::endl;
        return constructResp(false, chunk.localId, "Got an error");
    }
    std::cout << "GOT E TENSOR" << std::endl;
    std::cout << "EDGE INFO: " << eInfo.numLvids << ", " << eInfo.nChunkEdges << std::endl;

    std::vector<std::string> dataRequests{"z"};
    std::vector<Matrix> matrices = reqTensors(data_socket, chunk, dataRequests);
    for (auto& M : matrices) {
        if (M.empty()){
            std::cout << M.name() << " is empty" << std::endl;
            return constructResp(false, chunk.localId, M.name() + " is empty");
        }
    }
    std::cout << "GOT DATA" << std::endl;

    std::vector<std::string> weightRequests{"a_i"};
    std::vector<Matrix> weights = reqTensors(weights_socket, chunk, weightRequests);
    for (auto& W : weights) {
        if (W.empty()){
            std::cout << W.name() << " is empty" << std::endl;
            return constructResp(false, chunk.localId, W.name() + " is empty");
        }
    }
    std::cout << "GOT WEIGHTS" << std::endl;

    if (matrices.empty() || weights.empty()) {
        return constructResp(false, chunk.localId, "Got error message from server");
    }

    Matrix& Z = matrices[0];
    Matrix& a = weights[0];

    std::cout << Z.shape() << std::endl;
    std::cout << a.shape() << std::endl;

    Matrix edgeValInputs = edgeMatMul(eInfo, Z, a);
    deleteMatrix(Z);
    deleteMatrix(a);
    deleteEdgeInfo(eInfo);
    edgeValInputs.setName("az");

    Matrix edgeVals = leakyReLU(edgeValInputs);
    edgeVals.setName("A");

    std::vector<Matrix> toSend = {edgeVals, edgeValInputs};
    int ret = sendEdgeTensors(data_socket, chunk, toSend, true);

    deleteMatrix(edgeValInputs);
    deleteMatrix(edgeVals);

    return constructResp(true, chunk.localId, "Finished apply edge");
}

invocation_response
apply_edge_backward(zmq::socket_t& data_socket, zmq::socket_t& weights_socket, Chunk &chunk) {
    std::cout << "BACKWARD APPLY EDGE LAYER " << chunk.layer << std::endl;

    unsigned startRequest = timestamp_ms();
    EdgeInfo eTensor = reqEdgeInfo(data_socket, chunk);
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
    } else if (eTensor.numLvids == ERR_HEADER_FIELD) {
        std::cerr << "Prorably a null pointer? I dunno" << std::endl;
        return constructResp(false, chunk.localId, "Got an error");
    }

    std::vector<std::string> dataRequests{"z", "grad"};
    std::vector<Matrix> matrices = reqTensors(data_socket, chunk, dataRequests);
    for (auto& M : matrices) {
        if (M.empty()){
            std::cout << M.name() << " is empty" << std::endl;
            return constructResp(false, chunk.localId, M.name() + " is empty");
        }
    }

    Matrix z = matrices[0];
    Matrix grad = matrices[1];
    Matrix az = reqEdgeTensor(data_socket, chunk, "az");

    std::vector<std::string> weightRequests{"a_i"};
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

    Matrix &a = weights[0];

    // std::cerr << "z " << z.shape() << std::endl;
    // std::cerr << "azv " << azv.shape() << std::endl;
    // std::cerr << "a " << a.shape() << std::endl;
    // Matrix az = expand(azv, eTensor);
    // azv.free();
    // std::cerr << "az " << az.shape() << std::endl;

    Matrix dLRelu = leakyReLUDerivative(az);
    az.free();
    // std::cerr << "dLRelu " << dLRelu.shape() << std::endl;
    Matrix dAct = expandHadamardMul(grad, dLRelu, eTensor);
    // std::cerr << "dAct " << dAct.shape() << std::endl;
    grad.free();
    dLRelu.free();

    Matrix dA = dAct.dot(a);
    dA.setName("dA");
    // std::cerr << "dA " << dA.shape() << std::endl;

    std::vector<Matrix> toSend = { dA };
    // std::cerr << "Sending dA tensor" << std::endl;
    sendEdgeTensors(data_socket, chunk, toSend, true);
    // std::cerr << "Fin send" << std::endl;
    dA.free();

    Matrix dAct_reduce = reduce(dAct);
    dAct.free();

    Matrix zz = z.dot(z, true, false);
    z.free();
    Matrix da = zz.dot(dAct_reduce, false, true);
    dAct_reduce.free();
    zz.free();

    da.setName("a_i");
    std::vector<Matrix> weightUpds = { da };
    // std::cerr << "Sending weight upd" << std::endl;
    sendTensors(weights_socket, chunk, weightUpds);
    da.free();

    for (auto &w : weights) {
        w.free();
    }
    deleteEdgeInfo(eTensor);

    return constructResp(true, chunk.localId, "Finished apply edge backward");
}

invocation_response
apply_vertex(zmq::socket_t& data_socket, zmq::socket_t& weights_socket, Chunk &chunk) {
    std::cout << "FORWARD APPLY VERTEX LAYER " << chunk.layer << std::endl;

    std::vector<Matrix> matrices;
    // Request H directly
    std::vector<std::string> dataRequests{"h"};
    matrices = reqTensors(data_socket, chunk, dataRequests);
    for (auto& M : matrices) {
        if (M.empty()){
            std::cout << M.name() << " is empty" << std::endl;
            return constructResp(false, chunk.localId, M.name() + " is empty");
        }
    }

    Matrix& H = matrices[0];
    // std::cerr << "Get H " << H.shape() << std::endl;
    std::vector<std::string> weightRequests{"w"};
    std::vector<Matrix> weights = reqTensors(weights_socket, chunk, weightRequests);
    for (auto& W : weights) {
        if (W.empty()){
            H.free();
            std::cout << W.name() << " is empty" << std::endl;
            return constructResp(false, chunk.localId, W.name() + " is empty");
        }
    }

    if (matrices.empty() || weights.empty()) {
        return constructResp(false, chunk.localId, "Got error message from server");
    }

    Matrix& W = weights[0];
    // std::cerr << "Get W " << W.shape() << std::endl;

    Matrix Z = H.dot(W);
    Z.setName("z");
    deleteMatrix(H);
    deleteMatrix(W);

    std::vector<Matrix> toSend = {Z};
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

invocation_response
apply_vertex_backward(zmq::socket_t& data_socket, zmq::socket_t& weights_socket, Chunk &chunk) {
    std::cout << "BACKWARD APPLY VERTEX LAYER " << chunk.layer << std::endl;

    Matrix H;
    Matrix grad;
    std::vector<Matrix> matrices;

    // Request H directly
    std::vector<std::string> dataRequests{"h", "aTg"};
    matrices = reqTensors(data_socket, chunk, dataRequests);
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

    H = matrices[0];
    grad = matrices[1];
    Matrix& W = weights[0];

    if (chunk.layer != 0) {
        Matrix resultsGrad = grad.dot(W, false, true);
        resultsGrad.setName("grad");
        deleteMatrix(W);

        std::vector<Matrix> toSend;
        toSend.push_back(resultsGrad);
        std::cout << "Sending grad tensor" << std::endl;
        sendTensors(data_socket, chunk, toSend, true);
        std::cout << "Fin send" << std::endl;
        deleteMatrix(resultsGrad);
    } else {
        sendFinMsg(data_socket, chunk);
    }

    Matrix wUpd = H.dot(grad, true, false);
    wUpd.setName("w");
    deleteMatrix(H);
    deleteMatrix(grad);

    std::vector<Matrix> weightUpds { wUpd };
    sendTensors(weights_socket, chunk, weightUpds);

    return constructResp(true, chunk.localId, "Finished backward layer");
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
    auto identity = constructIdentity(chunk);

    zmq::socket_t weights_socket(ctx, ZMQ_DEALER);
    zmq::socket_t data_socket(ctx, ZMQ_DEALER);
    std::cout << "Setting up comms" << std::endl;
    try {
        weights_socket.setsockopt(ZMQ_IDENTITY, identity.data(), identity.size());
        if (RESEND) {
            weights_socket.setsockopt(ZMQ_RCVTIMEO, TIMEOUT_PERIOD);
        }
        char whost_port[50];
        sprintf(whost_port, "tcp://%s:%u", weightserver.c_str(), wport);
        weights_socket.connect(whost_port);

        data_socket.setsockopt(ZMQ_IDENTITY, identity.data(), identity.size());
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

    std::cout << chunk.str() << std::endl;
    if (chunk.vertex && chunk.dir == PROP_TYPE::FORWARD) {
        return apply_vertex(data_socket, weights_socket, chunk);
    } else if (chunk.vertex && chunk.dir == PROP_TYPE::BACKWARD) {
        return apply_vertex_backward(data_socket, weights_socket, chunk);
    } else if (!chunk.vertex && chunk.dir == PROP_TYPE::FORWARD) {
        return apply_edge(data_socket, weights_socket, chunk);
    } else { // !chunk.vertex && chunk.dir == PROP_TYPE::BACKWARD
        return apply_edge_backward(data_socket, weights_socket, chunk);
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


// invocation_response
// apply_edge_backward(zmq::socket_t& data_socket, zmq::socket_t& weights_socket, Chunk &chunk) {
//     std::cout << "BACKWARD APPLY EDGE LAYER " << chunk.layer << std::endl;

//     unsigned startRequest = timestamp_ms();
//     EdgeInfo eInfo = reqEdgeInfo(data_socket, chunk);
//     unsigned endRequest = timestamp_ms();
//     if (eInfo.numLvids == NOT_FOUND_ERR_FIELD) {
//         std::cerr << "Tensor 'fedge' was not found on graph server" << std::endl;
//         return constructResp(false, chunk.localId, "Tensor 'fedge' not found");
//     } else if (eInfo.numLvids == DUPLICATE_REQ_ERR_FIELD) {
//         std::cerr << "Chunk already running. Request rejected" << std::endl;
//         return constructResp(false, chunk.localId, "Duplicate chunk request");
//     } else if (eInfo.numLvids == CHUNK_DNE_ERR) {
//         std::cerr << "Chunk not found on graph server" << std::endl;
//         return constructResp(false, chunk.localId, "Chunk not found");
//     } else if (eInfo.numLvids == ERR_HEADER_FIELD) {
//         std::cerr << "Prorably a null pointer? I dunno" << std::endl;
//         return constructResp(false, chunk.localId, "Got an error");
//     }

//     std::cout << "EDGE INFO: " << eInfo.numLvids << ", " << eInfo.nChunkEdges << std::endl;

//     Matrix aZ = reqEdgeTensor(data_socket, chunk, "az");

//     std::vector<std::string> dataRequests{"z", "grad"};
//     std::vector<Matrix> matrices = reqTensors(data_socket, chunk, dataRequests);
//     Matrix& Z = matrices[0];
//     Matrix& dP = matrices[1];

//     std::cout << "Z: " << Z.shape() << std::endl;
//     std::cout << "dP: " << dP.shape() << std::endl;
//     std::cout << "aZ: " << aZ.shape() << std::endl;

//     std::vector<std::string> weightRequests{"a_i"};
//     std::vector<Matrix> weights = reqTensors(weights_socket, chunk, weightRequests);
//     Matrix& a = weights[0];
//     std::cout << "a: " << a.shape() << std::endl;

//     return constructResp(true, chunk.localId, "Finished apply edge backward");
// }
