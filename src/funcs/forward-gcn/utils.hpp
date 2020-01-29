#ifndef __FUNCS_UTILS__
#define __FUNCS_UTILS__

#include <chrono>
#include "../../../src/common/matrix.hpp"

extern unsigned timestamps[30];
extern unsigned tsidx;

// Timestamping operations
void set_timestamp() {
    auto now = system_clock::now().time_since_epoch();
    unsigned me = duration_cast<milliseconds>(now).count() - BASE_TMSP;
    timestamps[tsidx++] = me;
}

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
    socket.recv(&respHeader);
    // while (!socket.recv(&respHeader, ZMQ_DONTWAIT)) {
    //     usleep(SLEEP_PERIOD);
    //     if (reqTimer.peek() > TIMEOUT_PERIOD) {
    //         // failed
    //         // zmq::message_t _hdr(HEADER_SIZE);
    //         // populateHeader((char *) _hdr.data(), op, id);
    //         zmq::message_t _hdr;
    //         _hdr.copy(&header);
    //         socket.send(_hdr);
    //         reqTimer.start();
    //     }
    // }

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

        // printf("recv mat (%u, %u)\n", rows, cols);

        if (data) {
            unsigned recv_ts = *((unsigned *)respHeader.data() + 5);
            timestamps[tsidx++] = recv_ts;
            unsigned send_ts = *((unsigned *)respHeader.data() + 6);
            timestamps[tsidx++] = send_ts;
        }

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
    socket.recv(&confirm);
    // while(!socket.recv(&confirm, ZMQ_DONTWAIT)) {
    //     usleep(SLEEP_PERIOD);
    //     if (sndTimer.peek() > TIMEOUT_PERIOD) {
    //         zmq::message_t _hdr;
    //         _hdr.copy(&header);
    //         socket.send(header, ZMQ_SNDMORE);
    //         // zmq::message_t _updMsg(matrix.getDataSize());
    //         // std::memcpy(_updMsg.data(), matrix.getData(), matrix.getDataSize());
    //         zmq::message_t _zDt;
    //         _zDt.copy(&zData);
    //         socket.send(_zDt, ZMQ_SNDMORE);
    //         zmq::message_t _actDt;
    //         _actDt.copy(&actData);
    //         socket.send(_actDt);
    //         sndTimer.start();
    //     }
    // }

    {
        unsigned recv_ts = *((unsigned *)confirm.data());
        timestamps[tsidx++] = recv_ts;
        unsigned send_ts = *((unsigned *)confirm.data() + 1);
        timestamps[tsidx++] = send_ts;
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

        FeatType denom = 1e-20;
        FeatType maxEle = *(std::max_element(vecSrc, vecSrc + length));
        for (unsigned c = 0; c < length; ++c) {
            vecDst[c] = std::exp(vecSrc[c] - maxEle);
            denom += vecDst[c];
        }

        for (unsigned c = 0; c < length; ++c) {
            vecDst[c] /= denom;
        }
    }

    return Matrix(mat.getRows(), mat.getCols(), result);
}


// temp deprecated. Since we are going to calc acc and loss on graph servers
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



#endif // __FUNCS_UTILS__