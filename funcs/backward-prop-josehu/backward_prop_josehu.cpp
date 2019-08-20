#include <chrono>
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
#include "../../src/utils/utils.hpp"


#define LEARNING_RATE 0.1


#define SND_MORE true
#define NO_MORE false


using boost::property_tree::ptree;
using boost::property_tree::read_json;
using boost::property_tree::write_json;
using namespace aws::lambda_runtime;
using namespace std::chrono;


/** Struct for wrapping over the returned matrices. */
typedef struct {
    std::vector<Matrix> zMatrices;          // Layer 1 -> last.
    std::vector<Matrix> actMatrices;        // Layer 0 -> last.
    Matrix targetMatrix;
} GraphData;


/**
 *
 * Request the graph feature matrices data from dataserver.
 * 
 */
static GraphData
requestFeatsMatrices(zmq::socket_t& socket, unsigned id, unsigned numLayers) {
    
    // Send pull request.
    zmq::message_t header(HEADER_SIZE);
    populateHeader((char *) header.data(), OP::PULL_BACKWARD, id);
    socket.send(header);

    GraphData graphData;

    // Receive z matrices chunks, from layer 1-> last.
    for (size_t i = 1; i <= numLayers; ++i) {
        zmq::message_t respHeader;
        socket.recv(&respHeader);
        unsigned layerResp = parse<unsigned>((char *) respHeader.data(), 1);
        if (layerResp == ERR_HEADER_FIELD) {    // Failed.
            std::cerr << "[ ERROR ] No corresponding matrix chunk!" << std::endl;
            return graphData;
        } else {                    // Get matrices data.
            unsigned rows = parse<unsigned>((char *) respHeader.data(), 2);
            unsigned cols = parse<unsigned>((char *) respHeader.data(), 3);
            zmq::message_t matxData(rows * cols * sizeof(FeatType));
            socket.recv(&matxData);

            FeatType *matxBuffer = new FeatType[rows * cols];
            std::memcpy(matxBuffer, matxData.data(), matxData.size());

            graphData.zMatrices.push_back(Matrix(rows, cols, matxBuffer));
        }
    }

    // Receive act matrices chunks, from layer 0 -> last.
    for (size_t i = 0; i <= numLayers; ++i) {
        zmq::message_t respHeader;
        socket.recv(&respHeader);
        unsigned layerResp = parse<unsigned>((char *) respHeader.data(), 1);
        if (layerResp == ERR_HEADER_FIELD) {    // Failed.
            std::cerr << "[ ERROR ] No corresponding matrix chunk!" << std::endl;
            return graphData;
        } else {                    // Get matrices data.
            unsigned rows = parse<unsigned>((char *) respHeader.data(), 2);
            unsigned cols = parse<unsigned>((char *) respHeader.data(), 3);
            zmq::message_t matxData(rows * cols * sizeof(FeatType));
            socket.recv(&matxData);

            FeatType *matxBuffer = new FeatType[rows * cols];
            std::memcpy(matxBuffer, matxData.data(), matxData.size());

            graphData.actMatrices.push_back(Matrix(rows, cols, matxBuffer));
        }
    }

    // Receive target label matrix chunk.
    zmq::message_t respHeader;
    socket.recv(&respHeader);
    unsigned layerResp = parse<unsigned>((char *) respHeader.data(), 1);
    if (layerResp == ERR_HEADER_FIELD) {    // Failed.
        std::cerr << "[ ERROR ] No corresponding matrix chunk!" << std::endl;
        return graphData;
    } else {                    // Get matrices data.
        unsigned rows = parse<unsigned>((char *) respHeader.data(), 2);
        unsigned cols = parse<unsigned>((char *) respHeader.data(), 3);
        zmq::message_t matxData(rows * cols * sizeof(FeatType));
        socket.recv(&matxData);

        FeatType *matxBuffer = new FeatType[rows * cols];
        std::memcpy(matxBuffer, matxData.data(), matxData.size());

        graphData.targetMatrix = Matrix(rows, cols, matxBuffer);
    }

    return graphData;
}


/**
 *
 * Request the weight matrices data from weightserver.
 * 
 */
static std::vector<Matrix>
requestWeightsMatrices(zmq::socket_t& socket, unsigned layer) {
    std::vector<Matrix> weightsData;

    // TODO!
    FeatType *w2Data = new FeatType[10 * 4] {
        0.170054, -0.901535, 0.908919, 0.194343,
        -0.674725, -0.105672, -0.0228593, -0.198091,
        0.685961, -0.0546585, -0.646025, -0.733765,
        0.609123, -1.47173, -0.315495, 1.48348,
        -0.210511, -1.06066, -0.483713, 0.234035,
        0.420494, -0.757625, -1.40361, 1.47215,
        1.44152, -0.354305, 0.190449, -0.114473,
        -0.955115, 0.389505, 0.408238, 0.265304,
        0.970574, 1.43, 0.952564, -1.24768,
        0.221666, -0.46031, 0.575747, -1.4179
    };
    weightsData.push_back(Matrix(10, 4, w2Data));

    return weightsData;
}


/**
 *
 * Send weight updates back to weightserver.
 * 
 */
static void
sendWeightsUpdates(std::vector<Matrix> weightsUpdates, zmq::socket_t& socket, unsigned id) {
    // TODO!
    std::cout << "!!! Weights upates: ";
    for (Matrix& mat : weightsUpdates)
        std::cout << "(" << mat.getRows() << ", " << mat.getCols() << ") ";
    std::cout << std::endl;
}


/**
 *
 * Send finish message back to dataserver.
 * 
 */
static void
sendFinishMsg(zmq::socket_t& socket, unsigned id) {
    
    // Send push header.
    zmq::message_t header(HEADER_SIZE);
    populateHeader((char *) header.data(), OP::PUSH_BACKWARD, id);
    socket.send(header);

    // Wait for message confirmed reply.
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
    FeatType *res = new FeatType[mat.getRows() * mat.getCols()];

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
 * Element-wise subtraction & multiplication. MUST ensure inputs are in the same size.
 * 
 */
static Matrix
hadamardSub(Matrix& matLeft, Matrix& matRight) {
    assert(matLeft.getRows() == matRight.getRows());
    assert(matLeft.getCols() == matRight.getCols());
    
    FeatType *res = new FeatType[matLeft.getRows() * matLeft.getCols()];
    FeatType *leftData = matLeft.getData(), *rightData = matRight.getData();

    for (unsigned i = 0; i < matLeft.getRows() * matLeft.getCols(); ++i)
        res[i] = leftData[i] - rightData[i];

    return Matrix(matLeft.getRows(), matRight.getCols(), res);
}

static Matrix
hadamardMul(Matrix& matLeft, Matrix& matRight) {
    assert(matLeft.getRows() == matRight.getRows());
    assert(matLeft.getCols() == matRight.getCols());
    
    FeatType *res = new FeatType[matLeft.getRows() * matLeft.getCols()];
    FeatType *leftData = matLeft.getData(), *rightData = matRight.getData();

    for (unsigned i = 0; i < matLeft.getRows() * matLeft.getCols(); ++i)
        res[i] = leftData[i] * rightData[i];

    return Matrix(matLeft.getRows(), matRight.getCols(), res);
}


/**
 *
 * Apply derivate of the activation function on a matrix.
 * 
 */
static Matrix
activateDerivate(Matrix& mat) {
    FeatType *res = new FeatType[mat.getRows() * mat.getCols()];
    FeatType *zData = mat.getData();

    for (unsigned i = 0; i < mat.getRows() * mat.getCols(); ++i)
        res[i] = 1 - std::pow(std::tanh(zData[i]), 2);

    return Matrix(mat.getRows(), mat.getCols(), res);
}


/**
 *
 * Matrix multiplication functions used.
 * 
 */
static Matrix
dotGDwithWTrans(Matrix& matLeft, Matrix& matRight) {
    unsigned m = matLeft.getRows(), k = matLeft.getCols(), n = matRight.getRows();
    assert(k == matRight.getCols());

    FeatType *res = new FeatType[m * n];
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, 1.0,
                matLeft.getData(), k, matRight.getData(), k, 0.0, res, n);

    return Matrix(m, n, res);
}

static Matrix
dotActTranswithGD(Matrix& matLeft, Matrix& matRight, float alpha) {
    unsigned m = matLeft.getCols(), k = matLeft.getRows(), n = matRight.getCols();
    assert(k == matRight.getRows());

    FeatType *res = new FeatType[m * n];
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, m, n, k, alpha,
                matLeft.getData(), m, matRight.getData(), n, 0.0, res, n);

    return Matrix(m, n, res);
}


/**
 *
 * Main logic of gradient computation and a naive gradient descent to get weight updates.
 *
 * Attention:
 *   zMatrices   vec contains z1   -> zout;
 *   actMatrices vec contains act0 -> actout;
 *   weightData  vec contains w2   -> wout.
 * 
 */
static std::vector<Matrix>
gradientComputation(GraphData& graphData, std::vector<Matrix>& weightsData) {
    
    std::vector<Matrix> gradients;
    std::vector<Matrix> weightsUpdates;

    // Compute last layer's gradients.
    Matrix softmaxRes = softmaxRows(graphData.actMatrices.back());
    Matrix subRes = hadamardSub(softmaxRes, graphData.targetMatrix);
    Matrix derivateRes = activateDerivate(graphData.zMatrices.back());
    gradients.push_back(hadamardMul(subRes, derivateRes));
    delete[] softmaxRes.getData();
    delete[] subRes.getData();
    delete[] derivateRes.getData();

    // Compute previous layers gradients.
    for (unsigned i = weightsData.size(); i > 0; --i) {
        Matrix dotRes = dotGDwithWTrans(gradients.back(), weightsData[i - 1]);
        Matrix derivateRes = activateDerivate(graphData.zMatrices[i - 1]);
        gradients.push_back(hadamardMul(dotRes, derivateRes));
        delete[] dotRes.getData();
        delete[] derivateRes.getData();
    }

    std::reverse(gradients.begin(), gradients.end());

    // Compute weights updates.
    for (unsigned i = 0; i < gradients.size(); ++i) {
        weightsUpdates.push_back(dotActTranswithGD(graphData.actMatrices[i], gradients[i], LEARNING_RATE));
        delete[] gradients[i].getData();
    }

    return weightsUpdates;
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
              unsigned id, unsigned numLayers) {
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
    Timer gradientTimer;
    Timer sendResTimer;

    try {
        GraphData graphData;
        std::vector<Matrix> weightsData;     // Layer 2 -> last.
        std::vector<Matrix> weightsUpdates;

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

        // Request weights matrices.
        std::thread t([&] {     // Weight requests run in a separate thread.
            std::cout << "< BACKWARD > Asking weightserver..." << std::endl;
            getWeightsTimer.start();
            weightsData = requestWeightsMatrices(weights_socket, numLayers);
            getWeightsTimer.stop();
            std::cout << "< BACKWARD > Got data from weightserver." << std::endl;
        });

        // Request z, act & target matrices chunks.
        std::cout << "< BACKWARD > Asking dataserver..." << std::endl;
        getFeatsTimer.start();
        graphData = requestFeatsMatrices(data_socket, id, numLayers);
        getFeatsTimer.stop();
        std::cout << "< BACKWARD > Got data from dataserver." << std::endl;

        t.join();

        // if (weightsData.empty())
        //     return invocation_response::failure("Weights could not be loaded", "application/json");
        if (graphData.zMatrices.empty())
            return invocation_response::failure("No chunks corresponding to request", "appliation/json");

        // Gradient computation.
        gradientTimer.start();
        std::cout << "< BACKWARD > Doing the gradient descent computation..." << std::endl;
        weightsUpdates = gradientComputation(graphData, weightsData);
        gradientTimer.stop();

        // Send weight updates to weightserver, and finish message to dataserver.
        sendResTimer.start();
        std::cout << "< BACKWARD > Sending weight updates back..." << std::endl;
        sendWeightsUpdates(weightsUpdates, weights_socket, id);
        std::cout << "< BACKWARD > Weight updates sent." << std::endl;
        sendFinishMsg(data_socket, id);
        std::cout << "< BACKWARD > Finish message sent." << std::endl;
        sendResTimer.stop();

        // Delete malloced spaces.
        for (Matrix& mat : weightsData)
            delete[] mat.getData();
        for (Matrix& mat : graphData.zMatrices)
            delete[] mat.getData();
        for (Matrix& mat : graphData.actMatrices)
            delete[] mat.getData();
        delete[] graphData.targetMatrix.getData();
        for (Matrix& mat : weightsUpdates)
            delete[] mat.getData();

    } catch(std::exception &ex) {
        return invocation_response::failure(ex.what(), "application/json");
    }

    // Couldn't parse JSON with AWS SDK from ptree.
    // For now creating a string with the times to be parsed on server.
    std::string res = std::to_string(id) + ": " +
                      std::to_string(getWeightsTimer.getTime()) + " " +     \
                      std::to_string(getFeatsTimer.getTime())  + " " +      \
                      std::to_string(gradientTimer.getTime()) + " " +       \
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
    unsigned numLayers = pt.get<int>("layer");
    unsigned chunkId = pt.get<int>("id");

    std::cout << "[ACCEPTED] Thread " << chunkId << " is requested from " << dataserver << ":" << dport
              << ", BACKWARD on " << numLayers << " layers." << std::endl;

    return backward_prop(dataserver, weightserver, dport, wport, chunkId, numLayers);
}

int
main(int argc, char *argv[]) {
    run_handler(my_handler);
    
    return 0;
}
