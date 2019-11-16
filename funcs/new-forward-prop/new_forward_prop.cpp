#include <algorithm>
#include <chrono>
#include <ratio>
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

#include "../../src/common/matrix.hpp"
#include "../../src/common/utils.hpp"

#define SLEEP_PERIOD   1000  // us
#define TIMEOUT_PERIOD (500) // ms

#define SND_MORE true
#define NO_MORE false


using boost::property_tree::ptree;
using boost::property_tree::read_json;
using boost::property_tree::write_json;
using namespace aws::lambda_runtime;
using namespace std::chrono;

bool lastLayer = false;


/**
 *
 * Request the input matrix data from dataserver.
 *
 */
static Matrix
requestMatrix(zmq::socket_t& socket, OP op, unsigned id, bool data = false) {
    // Send pull request.
    zmq::message_t header(HEADER_SIZE);
    populateHeader((char *) header.data(), op, id);
    socket.send(header);

    Timer reqTimer;
    reqTimer.start();

    // Listen on respond.
    zmq::message_t respHeader;
    // socket.recv(&respHeader);
    while (!socket.recv(&respHeader, ZMQ_DONTWAIT)) {
        usleep(SLEEP_PERIOD);
        if (reqTimer.peek() > TIMEOUT_PERIOD) {
            // failed
            // zmq::message_t _hdr(HEADER_SIZE);
            // populateHeader((char *) _hdr.data(), op, id);
            zmq::message_t _hdr;
            _hdr.copy(&header);
            socket.send(_hdr);
            reqTimer.start();
        }
    }

    // Parse the respond.
    unsigned layerResp = parse<unsigned>((char *) respHeader.data(), 1);
    if (layerResp == -2) {
        std::cerr << "[ ERROR ] Discard execution." << std::endl;
        exit(0);
    } else if (layerResp == -1) {      // Failed.
        std::cerr << "[ ERROR ] No corresponding matrix chunk!" << std::endl;
        return Matrix();
    } else {                    // Get matrix data.
        unsigned rows = parse<unsigned>((char *) respHeader.data(), 2);
        unsigned cols = parse<unsigned>((char *) respHeader.data(), 3);

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
 * Send multiplied matrix result back to dataserver.
 *
 */
static void
sendMatrices(Matrix& zResult, Matrix& actResult, zmq::socket_t& socket, unsigned id) {

    // Send push header.
    zmq::message_t header(HEADER_SIZE);
    populateHeader((char *) header.data(), OP::PUSH_FORWARD, id, zResult.getRows(), zResult.getCols());
    socket.send(header, ZMQ_SNDMORE);

    // Send zData and actData.
    zmq::message_t zData(zResult.getDataSize());
    std::memcpy(zData.data(), zResult.getData(), zResult.getDataSize());
    zmq::message_t actData(actResult.getDataSize());
    std::memcpy(actData.data(), actResult.getData(), actResult.getDataSize());
    socket.send(zData, ZMQ_SNDMORE);
    socket.send(actData);

    Timer sndTimer;
    sndTimer.start();

    // Wait for data settled reply.
    zmq::message_t confirm;
    // socket.recv(&confirm);
    while(!socket.recv(&confirm, ZMQ_DONTWAIT)) {
        usleep(SLEEP_PERIOD);
        if (sndTimer.peek() > TIMEOUT_PERIOD) {
            zmq::message_t _hdr;
            _hdr.copy(&header);
            socket.send(header, ZMQ_SNDMORE);
            // zmq::message_t _updMsg(matrix.getDataSize());
            // std::memcpy(_updMsg.data(), matrix.getData(), matrix.getDataSize());
            zmq::message_t _zDt;
            _zDt.copy(&zData);
            socket.send(_zDt, ZMQ_SNDMORE);
            zmq::message_t _actDt;
            _actDt.copy(&actData);
            socket.send(_actDt);
            sndTimer.start();
        }
    }
}

static void
sendWeightUpdates(zmq::socket_t& socket, Matrix& weightUpdates, unsigned layer) {
    zmq::message_t header(HEADER_SIZE);
    populateHeader((char*) header.data(), OP::PUSH_BACKWARD, layer);
    socket.send(header, ZMQ_SNDMORE);

    zmq::message_t updateMsg(weightUpdates.getDataSize());
    std::memcpy(updateMsg.data(), weightUpdates.getData(), weightUpdates.getDataSize());
    socket.send(updateMsg);

    zmq::message_t confirm;
    socket.recv(&confirm);
}


/**
 *
 * Matrix multiplication function.
 *
 */
static Matrix
dot(Matrix& features, Matrix& weights) {
    unsigned m = features.getRows(), k = features.getCols(), n = weights.getCols();
    assert(k == weights.getRows());

    FeatType *res = new FeatType[m * n];
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0,
                features.getData(), k, weights.getData(), n, 0.0, res, n);

    return Matrix(m, n, res);
}


/**
 *
 * Apply activation function on a matrix.
 *
 */
static Matrix
activate(Matrix& mat) {
    FeatType *activationData = new FeatType[mat.getNumElemts()];
    FeatType *zData = mat.getData();

    for (unsigned i = 0; i < mat.getNumElemts(); ++i)
        activationData[i] = std::tanh(zData[i]);

    return Matrix(mat.getRows(), mat.getCols(), activationData);
}

/**
 *
 * Apply softmax to all rows of the input matrix (Currently overwrites the
 * input matrix data)
 *
 */
static Matrix
softmax(Matrix& mat) {
    FeatType* result = new FeatType[mat.getNumElemts()];

    for (unsigned r = 0; r < mat.getRows(); ++r) {
        unsigned length = mat.getCols();
        FeatType* vecSrc = mat.getData() + r * length;
        FeatType* vecDst = result + r * length;

        FeatType denom = 0.0;
        for (unsigned c = 0; c < length; ++c) {
            vecDst[c] = std::exp(vecSrc[c]);
            denom += vecDst[c];
        }

        for (unsigned c = 0; c < length; ++c) {
            vecDst[c] /= denom;
        }
    }

    return Matrix(mat.getRows(), mat.getCols(), result);
}

static unsigned
getMaxIndex(FeatType* row, unsigned length) {
    float max = 0.0;
    unsigned maxIndex = 0;
    for (unsigned col = 0; col < length; ++col) {
        if (row[col] > max) {
            max = row[col];
            maxIndex = col;
        }
    }

    return maxIndex;
}

static unsigned
getLabelIndex(FeatType* row, unsigned length) {
    for (unsigned col = 0; col < length; ++col) {
        if (row[col] == 1)
            return col;
    }

    // Should never get here
    return -1;
}

static unsigned
checkAccuracy(Matrix& predictions, Matrix& labels) {
    assert(predictions.getRows() == labels.getRows());
    assert(predictions.getCols() == labels.getCols());
    unsigned totalCorrect = 0;
    unsigned length = predictions.getCols();
    for (unsigned r = 0; r < predictions.getRows(); ++r) {
        unsigned maxIndex = getMaxIndex(predictions.get(r), length);

        if (labels.get(r, maxIndex) == 1.0)
            ++totalCorrect;
    }

    return totalCorrect;
}

static float
checkLoss(Matrix& preds, Matrix& labels) {
    assert(preds.getRows() == labels.getRows());
    assert(preds.getCols() == labels.getCols());

    float totalLoss = 0;
	unsigned length = preds.getCols();
    for (unsigned r = 0; r < preds.getRows(); ++r) {
        unsigned labelIndex = getLabelIndex(labels.get(r), length);
        // loss = -log(class_prediction)
        float lossThisRow = -(std::log(preds.get(r, labelIndex)));
        totalLoss += lossThisRow;
    }

    return totalLoss;
}

/**
 *
 * Evaluate the current state of the model using accuracy and loss
 *
 */
static void
evaluateModel(Matrix& activations, zmq::socket_t& datasocket, unsigned partId) {
    Matrix labels = requestMatrix(datasocket, OP::PULL_EVAL, partId);
    Matrix predictions = softmax(activations);

    // Check if the label with the highest probability after softmax is equal to the
    // target label
    unsigned totalCorrect = checkAccuracy(predictions, labels);

    // Sum the individual losses of each vertex for this validation partition
    float lossThisPart = checkLoss(predictions, labels);

    zmq::message_t header(HEADER_SIZE);
    populateHeader((char*)header.data(), OP::PUSH_EVAL, partId, totalCorrect);
    serialize<float>((char*)header.data(), 3, lossThisPart);

    datasocket.send(header);

    // Wait to recv ACK
    zmq::message_t confirm;
    datasocket.recv(&confirm);
}

static Matrix
gradLoss(Matrix& z, Matrix& weights, Matrix& AH, zmq::socket_t& datasocket, zmq::socket_t& weightSocket,
         unsigned partId, unsigned layer) {
    Matrix labels = requestMatrix(datasocket, OP::PULL_EVAL, partId);
    Matrix predictions = softmax(z);

    Matrix d_out = predictions - labels;

    // True indicating that AH is transposed
    // NOTE:
    //  In the future it would make more sense to return the d_AH
    //  values to the server first so that computation can proceed
    //  and then calculate the weight updates as there are no
    //  dependencies on the weight updates
    Matrix weightUpdates = AH.dot(d_out, true);

    sendWeightUpdates(weightSocket, weightUpdates, layer);

    // True here indicating the weights are transposed
    Matrix d_AH = d_out.dot(weights, false, true);

    return d_AH;
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
forward_prop_layer(std::string dataserver, std::string weightserver, std::string dport, std::string wport,
                   unsigned id, unsigned layer, bool lastLayer) {
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

    Timer getWeightsTimer;
    Timer getFeatsTimer;
    Timer computationTimer;
    Timer sendResTimer;

    try {
        Matrix weights, feats, z, activations;

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

        // Request weights matrix of the current layer.
        std::thread t([&] {     // Weight requests run in a separate thread.
            std::cout << "< FORWARD > Asking weightserver..." << whost_port << std::endl;
            getWeightsTimer.start();
            do {
                weights = requestMatrix(weights_socket, OP::PULL_FORWARD, layer);
            } while (weights.empty());
            getWeightsTimer.stop();
            std::cout << "< FORWARD > Got data from weightserver." << dhost_port << std::endl;
        });

        // Request feature activation matrix of the current layer.
        std::cout << "< FORWARD > Asking dataserver..." << std::endl;
        getFeatsTimer.start();
        do {
            feats = requestMatrix(data_socket, OP::PULL_FORWARD, id, true);
        } while (feats.empty());
        getFeatsTimer.stop();
        std::cout << "< FORWARD > Got data from dataserver." << std::endl;

        t.join();

        if (weights.empty())
            return invocation_response::failure("Weights could not be loaded", "application/json");
        if (feats.empty())
            return invocation_response::failure("No chunk corresponding to request", "appliation/json");


        // Multiplication.
        computationTimer.start();
        z = feats.dot(weights);
        computationTimer.stop();


        if (lastLayer) {
            activations = softmax(z);
        } else {
            activations = activate(z);
        }
        sendResTimer.start();
        sendMatrices(z, activations, data_socket, id);
        sendResTimer.stop();

        // Delete malloced spaces.
        delete[] weights.getData();
        delete[] feats.getData();
        delete[] z.getData();
        delete[] activations.getData();

    } catch(std::exception &ex) {
        return invocation_response::failure(ex.what(), "application/json");
    }

    // Couldn't parse JSON with AWS SDK from ptree.
    // For now creating a string with the times to be parsed on server.
    std::string res = std::to_string(id) + ": " +
                      std::to_string(getWeightsTimer.getTime()) + " " +     \
                      std::to_string(getFeatsTimer.getTime())  + " " +      \
                      std::to_string(computationTimer.getTime()) + " " +    \
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
    unsigned layer = pt.get<int>("layer");
    unsigned chunkId = pt.get<int>("id");
    lastLayer = pt.get<bool>("lastLayer");

    std::cout << "[ACCEPTED] Thread " << chunkId << " is requested from " << dataserver << ":" << dport
              << ", FORWARD layer " << layer << "." << std::endl;

    return forward_prop_layer(dataserver, weightserver, dport, wport, chunkId, layer, lastLayer);
}

int
main(int argc, char *argv[]) {
    run_handler(my_handler);

    return 0;
}
