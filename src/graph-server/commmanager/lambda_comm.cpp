#include "lambda_comm.hpp"
#include <thread>


std::mutex m;
std::condition_variable cv;
std::mutex count_mutex;


/**
 *
 * Lambdaworker is a wrapper over the sender & receiver thread.
 * 
 */
void
LambdaWorkerForward::work() {
    worker.connect("inproc://backend");

    try {
        while (true) {
            zmq::message_t identity;
            zmq::message_t header;

            worker.recv(&identity);
            worker.recv(&header);

            unsigned op = parse<unsigned>((char *) header.data(), 0);
            unsigned partId = parse<unsigned>((char *) header.data(), 1);

            switch (op) {
                case (OP::PULL):
                    sendMatrixChunk(worker, identity, partId);
                    break;
                case (OP::PUSH):
                    recvMatrixChunks(worker, identity, partId);
                    break;
                default:
                    printLog(nodeId, "WORKER ERROR: Unknown OP code (%u) received.\n", op);
            }
        }
    } catch (std::exception& ex) {
        printLog(nodeId, "WORKER ERROR: %s\n", ex.what());
    }
}


/**
 *
 * Sending & receiving matrix chunk to / from lambda threads.
 * 
 */
void
LambdaWorkerForward::sendMatrixChunk(zmq::socket_t& socket, zmq::message_t& client_id, unsigned partId) {
    zmq::message_t header(HEADER_SIZE);

    // Reject a send request if the partition id is invalid.
    if (partId >= numLambdas) {
        populateHeader((char *) header.data(), ERR_HEADER_FIELD, ERR_HEADER_FIELD, ERR_HEADER_FIELD, ERR_HEADER_FIELD);
        socket.send(client_id, ZMQ_SNDMORE);
        socket.send(header);

    // Partition id is valid, so send the matrix segment.
    } else {

        socket.send(client_id, ZMQ_SNDMORE);

        // Check to make sure that the bounds of this partition do not exceed the bounds of the data array.
        // If they do, set partition end to the end of the array.
        unsigned partRows = std::ceil((float) actMatrix->getRows() / (float) numLambdas);
        if ((partId * partRows + partRows) > actMatrix->getRows())
            partRows = partRows - (partId * partRows + partRows) + actMatrix->getRows();
        unsigned bufSize = partRows * actMatrix->getCols() * sizeof(FeatType);
        FeatType *partitionStart = actMatrix->getData() + (partId * partRows * actMatrix->getCols());

        populateHeader((char *) header.data(), OP::RESP, 0, partRows, actMatrix->getCols());
        socket.send(header, ZMQ_SNDMORE);

        zmq::message_t partitionData(bufSize);
        std::memcpy(partitionData.data(), partitionStart, bufSize);
        socket.send(partitionData);
    }
}

void
LambdaWorkerForward::recvMatrixChunks(zmq::socket_t& socket, zmq::message_t& client_id, unsigned partId) {
    unsigned partRows = std::ceil((float) actMatrix->getRows() / (float) numLambdas);
    FeatType *partitionZStart = zData + partId * partRows * numFeatsNext;
    FeatType *partitionActStart = actData + partId * partRows * numFeatsNext;

    // Receive the pushed-back results.
    zmq::message_t data;
    socket.recv(&data);
    std::memcpy(partitionZStart, data.data(), data.size());
    socket.recv(&data);
    std::memcpy(partitionActStart, data.data(), data.size());

    // Send confirm ACK message.
    zmq::message_t confirm;
    socket.send(client_id, ZMQ_SNDMORE);
    socket.send(confirm);

    // Check for total number of partitions received. If all partitions received, wake up lambdaComm.
    std::lock_guard<std::mutex> lock(count_mutex);
    ++count;
    if (count == numLambdas) {
        delete actMatrix;       // All chunks finished, so delete the matrix object here.
        cv.notify_one();
    }
}


/**
 *
 * Lambdaworker is a wrapper over the sender & receiver thread.
 * 
 */
void
LambdaWorkerBackward::work() {
    worker.connect("inproc://backend");

    try {
        while (true) {
            zmq::message_t identity;
            zmq::message_t header;

            worker.recv(&identity);
            worker.recv(&header);

            unsigned op = parse<unsigned>((char *) header.data(), 0);
            unsigned partId = parse<unsigned>((char *) header.data(), 1);

            switch (op) {
                case (OP::PULL):
                    sendMatrixChunks(worker, identity, partId);
                    break;
                default:
                    printLog(nodeId, "WORKER ERROR: Unknown Op code received.\n");
            }
        }
    } catch (std::exception& ex) {
        printLog(nodeId, "WORKER ERROR: %s\n", ex.what());
    }
}


/**
 *
 * Sending matrix chunks to lambda threads.
 * 
 */
void
LambdaWorkerBackward::sendMatrixChunks(zmq::socket_t& socket, zmq::message_t& client_id, unsigned partId) {
    zmq::message_t header(HEADER_SIZE);

    // Reject a send request if the partition id is invalid.
    if (partId >= numLambdas) {
        populateHeader((char *) header.data(), ERR_HEADER_FIELD, ERR_HEADER_FIELD, ERR_HEADER_FIELD, ERR_HEADER_FIELD);
        socket.send(client_id, ZMQ_SNDMORE);
        socket.send(header);

    // Partition id is valid, so send the matrix segments.
    } else {

        socket.send(client_id, ZMQ_SNDMORE);

        // Check to make sure that the bounds of this partition do not exceed the bounds of the data array.
        // If they do, set partition end to the end of the array.
        unsigned partRows = std::ceil((float) targetMatrix->getRows() / (float) numLambdas);
        if ((partId * partRows + partRows) > targetMatrix->getRows())
            partRows = partRows - (partId * partRows + partRows) + targetMatrix->getRows();

        // Send z matrices, from layer 1-> last.
        for (Matrix *matrix : zMatrices) {
            unsigned bufSize = partRows * matrix->getCols() * sizeof(FeatType);
            FeatType *partitionStart = matrix->getData() + (partId * partRows * matrix->getCols());

            populateHeader((char *) header.data(), OP::RESP, 0, partRows, matrix->getCols());
            socket.send(header, ZMQ_SNDMORE);

            zmq::message_t partitionData(bufSize);
            std::memcpy(partitionData.data(), partitionStart, bufSize);
            socket.send(partitionData, ZMQ_SNDMORE);
        }

        // Send activation matrices, from layer 0 -> last.
        for (Matrix *matrix : actMatrices) {
            unsigned bufSize = partRows * matrix->getCols() * sizeof(FeatType);
            FeatType *partitionStart = matrix->getData() + (partId * partRows * matrix->getCols());

            populateHeader((char *) header.data(), OP::RESP, 0, partRows, matrix->getCols());
            socket.send(header, ZMQ_SNDMORE);

            zmq::message_t partitionData(bufSize);
            std::memcpy(partitionData.data(), partitionStart, bufSize);
            socket.send(partitionData, ZMQ_SNDMORE);
        }

        // Send target label matrix.
        unsigned bufSize = partRows * targetMatrix->getCols() * sizeof(FeatType);
        FeatType *partitionStart = targetMatrix->getData() + (partId * partRows * targetMatrix->getCols());

        populateHeader((char *) header.data(), OP::RESP, 0, partRows, targetMatrix->getCols());
        socket.send(header, ZMQ_SNDMORE);

        zmq::message_t partitionData(bufSize);
        std::memcpy(partitionData.data(), partitionStart, bufSize);
        socket.send(partitionData);
    }
}


///////////////////////////////////
// Below are LambdaComm methods. //
///////////////////////////////////


/**
 *
 * Call 'newContextForward()' before the lambda invokation to refresh the parameters, and call 'endContextForward()' after
 * lambdas finish to revoke unused memory space (if necessary).
 * 
 */
void
LambdaComm::newContextForward(FeatType *dataBuf, FeatType *zData, FeatType *actData,
                              unsigned numLocalVertices, unsigned numFeats, unsigned numFeatsNext) {
    countForward = 0;

    // Create a new matrix object for workers to access.
    Matrix *actMatrix = new Matrix(numLocalVertices, numFeats, dataBuf);

    // Create 'numListeners' workers and detach them.
    for (unsigned i = 0; i < numListeners; ++i) {
        forwardWorkers.push_back(new LambdaWorkerForward(nodeId, ctx, numLambdasForward, countForward,
                                                         actMatrix, zData, actData, numFeatsNext));
        forwardWorker_threads.push_back(new std::thread(std::bind(&LambdaWorkerForward::work, forwardWorkers[i])));
        forwardWorker_threads[i]->detach();
    }

    printLog(nodeId, "Lambda FORWARD context created.\n");
}

void
LambdaComm::requestLambdasForward(unsigned layer) {

    // Send header info to tell the coordserver to trigger how many lambdas in which forward layer.
    zmq::message_t header(HEADER_SIZE);
    populateHeader((char *) header.data(), OP::REQ_FORWARD, layer, numLambdasForward);
    coordsocket.send(header, ZMQ_SNDMORE);

    // Send my ip.
    zmq::message_t ip_msg(nodeIp.size());
    std::memcpy(ip_msg.data(), nodeIp.c_str(), nodeIp.size());
    coordsocket.send(ip_msg);
    
    // Wait for a confirm ACK message.
    zmq::message_t confirm;
    coordsocket.recv(&confirm);

    // Block until all parts have been handled.
    std::unique_lock<std::mutex> lk(m);
    cv.wait(lk, [&]{ return countForward == numLambdasForward; });
}

void
LambdaComm::endContextForward() {
    countForward = 0;

    // Delete the workers.
    for (unsigned i = 0; i < numListeners; ++i) {
        delete forwardWorker_threads[i];
        delete forwardWorkers[i];
    }
    forwardWorkers.clear();
    forwardWorker_threads.clear();

    printLog(nodeId, "Lambda FORWARD context finished.\n");
}


/**
 *
 * Call 'newContextBackward()' before the lambda invokation to refresh the parameters, and call 'endContextBackward()' after
 * lambdas finish to revoke unused memory space (if necessary).
 * 
 */
void
LambdaComm::newContextBackward(std::vector<FeatType *> zBufs, std::vector<FeatType *> actBufs, FeatType *targetBuf,
                               unsigned numLocalVertices, std::vector<unsigned> layerConfig) {
    countBackward = 0;

    // Create new matrix objects for workers to access.
    std::vector<Matrix *> zMatrices;
    std::vector<Matrix *> actMatrices;
    Matrix *targetMatrix;

    assert(zBufs.size() == layerConfig.size());
    for (size_t i = 1; i < layerConfig.size(); ++i)
        zMatrices.push_back(new Matrix(numLocalVertices, layerConfig[i], zBufs[i]));

    assert(actBufs.size() == layerConfig.size());
    for (size_t i = 0; i < layerConfig.size(); ++i)
        actMatrices.push_back(new Matrix(numLocalVertices, layerConfig[i], actBufs[i]));

    targetMatrix = new Matrix(numLocalVertices, layerConfig[layerConfig.size() - 1], targetBuf);

    // Create 'numListeners' workers and detach them.
    for (unsigned i = 0; i < numListeners; ++i) {
        backwardWorkers.push_back(new LambdaWorkerBackward(nodeId, ctx, numLambdasBackward, countBackward,
                                                           actMatrices, zMatrices, targetMatrix));
        backwardWorker_threads.push_back(new std::thread(std::bind(&LambdaWorkerBackward::work, backwardWorkers[i])));
        backwardWorker_threads[i]->detach();
    }

    printLog(nodeId, "Lambda BACKWARD context created.\n");
}

void
LambdaComm::requestLambdasBackward() {

    // Send header info to tell the coordserver to trigger how many lambdas to trigger.
    zmq::message_t header(HEADER_SIZE);
    populateHeader((char *) header.data(), OP::REQ_BACKWARD, 0, numLambdasBackward);
    coordsocket.send(header, ZMQ_SNDMORE);

    // Send my ip.
    zmq::message_t ip_msg(nodeIp.size());
    std::memcpy(ip_msg.data(), nodeIp.c_str(), nodeIp.size());
    coordsocket.send(ip_msg);
    
    // Wait for a confirm ACK message.
    zmq::message_t confirm;
    coordsocket.recv(&confirm);

    // Block until all parts have been handled.
    std::unique_lock<std::mutex> lk(m);
    cv.wait(lk, [&]{ return countBackward == numLambdasBackward; });
}

void
LambdaComm::endContextBackward() {
    countBackward = 0;

    // Delete the workers.
    for (unsigned i = 0; i < numListeners; ++i) {
        delete backwardWorker_threads[i];
        delete backwardWorkers[i];
    }
    backwardWorkers.clear();
    backwardWorker_threads.clear();

    printLog(nodeId, "Lambda BACKWARD context finished.\n");
}


/**
 * 
 * Send message to the coordination server to shutdown.
 * 
 */
void
LambdaComm::sendShutdownMessage() {
    char chost_port[50];
    sprintf(chost_port, "tcp://%s:%u", coordserverIp.c_str(), coordserverPort);
    coordsocket.connect(chost_port);

    // Send kill message.
    zmq::message_t header(HEADER_SIZE);
    populateHeader((char *) header.data(), OP::TERM);
    coordsocket.send(header, ZMQ_SNDMORE);

    // Send dummy message since coordination server expects an IP as well.
    zmq::message_t dummyIP;
    coordsocket.send(dummyIP);
}
