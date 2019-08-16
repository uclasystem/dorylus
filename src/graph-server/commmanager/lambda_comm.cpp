#include "lambda_comm.hpp"
#include <thread>


std::mutex count_mutex;
std::condition_variable cv_forward;
std::mutex finish_mutex;
std::condition_variable cv_backward;


/**
 *
 * Lambdaworker is a wrapper over the sender & receiver thread.
 * 
 */
void
LambdaWorker::work() {
    try {
        while (true) {
            zmq::message_t identity;
            zmq::message_t header;

            workersocket.recv(&identity);
            workersocket.recv(&header);

            unsigned op = parse<unsigned>((char *) header.data(), 0);
            unsigned partId = parse<unsigned>((char *) header.data(), 1);

            switch (op) {
                case (OP::PULL_FORWARD):
                    sendAggregatedChunk(identity, partId);
                    break;
                case (OP::PUSH_FORWARD):
                    recvLambdaResults(identity, partId);
                    break;
                case (OP::PULL_BACKWARD):
                    sendBackpropChunks(identity, partId);
                    break;
                case (OP::PUSH_BACKWARD):
                    recvBackpropFinishMsg(identity);
                    break;
                default:
                    break;  /** Not an op that I care about. */
            }
        }
    } catch (std::exception& ex) { /** Context Termintated. */ }
}


/**
 *
 * Reset the member values for the next round of communication.
 * 
 */
void
LambdaWorker::refreshState(Matrix actMatrix_, FeatType *zData_, FeatType *actData_, unsigned numFeatsNext_) {           // For forward-prop.
    actMatrix = actMatrix_;
    zData = zData_;
    actData = actData_;
    numFeatsNext = numFeatsNext_;
}

void
LambdaWorker::refreshState(std::vector<Matrix> zMatrices_, std::vector<Matrix> actMatrices_, Matrix targetMatrix_) {    // For backward-prop.
    zMatrices = zMatrices_;
    actMatrices = actMatrices_;
    targetMatrix = targetMatrix_;
}


/**
 *
 * Sending & receiving messages to / from lambda threads.
 * 
 */
void
LambdaWorker::sendAggregatedChunk(zmq::message_t& client_id, unsigned partId) {
    zmq::message_t header(HEADER_SIZE);

    // Reject a send request if the partition id is invalid.
    if (partId >= numLambdasForward) {
        populateHeader((char *) header.data(), ERR_HEADER_FIELD, ERR_HEADER_FIELD, ERR_HEADER_FIELD, ERR_HEADER_FIELD);
        workersocket.send(client_id, ZMQ_SNDMORE);
        workersocket.send(header);

    // Partition id is valid, so send the matrix segment.
    } else {

        workersocket.send(client_id, ZMQ_SNDMORE);

        // Check to make sure that the bounds of this partition do not exceed the bounds of the data array.
        // If they do, set partition end to the end of the array.
        unsigned partRows = std::ceil((float) actMatrix.getRows() / (float) numLambdasForward);
        unsigned thisPartRows = partRows;
        if ((partId * partRows + partRows) > actMatrix.getRows())
            thisPartRows = partRows - (partId * partRows + partRows) + actMatrix.getRows();
        unsigned bufSize = thisPartRows * actMatrix.getCols() * sizeof(FeatType);
        FeatType *partitionStart = actMatrix.getData() + (partId * partRows * actMatrix.getCols());

        populateHeader((char *) header.data(), OP::RESP, 0, thisPartRows, actMatrix.getCols());
        workersocket.send(header, ZMQ_SNDMORE);

        zmq::message_t partitionData(bufSize);
        std::memcpy(partitionData.data(), partitionStart, bufSize);
        workersocket.send(partitionData);
    }
}

void
LambdaWorker::recvLambdaResults(zmq::message_t& client_id, unsigned partId) {
    unsigned partRows = std::ceil((float) actMatrix.getRows() / (float) numLambdasForward);
    FeatType *partitionZStart = zData + partId * partRows * numFeatsNext;
    FeatType *partitionActStart = actData + partId * partRows * numFeatsNext;

    // Receive the pushed-back results.
    zmq::message_t data;
    workersocket.recv(&data);
    std::memcpy(partitionZStart, data.data(), data.size());
    workersocket.recv(&data);
    std::memcpy(partitionActStart, data.data(), data.size());

    // Send confirm ACK message.
    zmq::message_t confirm;
    workersocket.send(client_id, ZMQ_SNDMORE);
    workersocket.send(confirm);

    // Check for total number of partitions received. If all partitions received, wake up lambdaComm.
    std::lock_guard<std::mutex> lk(count_mutex);
    ++countForward;
    if (countForward == numLambdasForward)
        cv_forward.notify_one();
}

void
LambdaWorker::sendBackpropChunks(zmq::message_t& client_id, unsigned partId) {
    zmq::message_t header(HEADER_SIZE);

    // Reject a send request if the partition id is invalid.
    if (partId >= numLambdasBackward) {
        populateHeader((char *) header.data(), ERR_HEADER_FIELD, ERR_HEADER_FIELD, ERR_HEADER_FIELD, ERR_HEADER_FIELD);
        workersocket.send(client_id, ZMQ_SNDMORE);
        workersocket.send(header);

    // Partition id is valid, so send the matrix segments.
    } else {

        workersocket.send(client_id, ZMQ_SNDMORE);

        // Check to make sure that the bounds of this partition do not exceed the bounds of the data array.
        // If they do, set partition end to the end of the array.
        unsigned partRows = std::ceil((float) targetMatrix.getRows() / (float) numLambdasBackward);
        unsigned thisPartRows = partRows;
        if ((partId * partRows + partRows) > targetMatrix.getRows())
            thisPartRows = partRows - (partId * partRows + partRows) + targetMatrix.getRows();

        // Send z matrices, from layer 1-> last.
        for (Matrix& matrix : zMatrices) {
            unsigned bufSize = thisPartRows * matrix.getCols() * sizeof(FeatType);
            FeatType *partitionStart = matrix.getData() + (partId * partRows * matrix.getCols());

            populateHeader((char *) header.data(), OP::RESP, 0, thisPartRows, matrix.getCols());
            workersocket.send(header, ZMQ_SNDMORE);

            zmq::message_t partitionData(bufSize);
            std::memcpy(partitionData.data(), partitionStart, bufSize);
            workersocket.send(partitionData, ZMQ_SNDMORE);
        }

        // Send activation matrices, from layer 0 -> last.
        for (Matrix& matrix : actMatrices) {
            unsigned bufSize = thisPartRows * matrix.getCols() * sizeof(FeatType);
            FeatType *partitionStart = matrix.getData() + (partId * partRows * matrix.getCols());

            populateHeader((char *) header.data(), OP::RESP, 0, thisPartRows, matrix.getCols());
            workersocket.send(header, ZMQ_SNDMORE);

            zmq::message_t partitionData(bufSize);
            std::memcpy(partitionData.data(), partitionStart, bufSize);
            workersocket.send(partitionData, ZMQ_SNDMORE);
        }

        // Send target label matrix.
        unsigned bufSize = thisPartRows * targetMatrix.getCols() * sizeof(FeatType);
        FeatType *partitionStart = targetMatrix.getData() + (partId * partRows * targetMatrix.getCols());

        populateHeader((char *) header.data(), OP::RESP, 0, thisPartRows, targetMatrix.getCols());
        workersocket.send(header, ZMQ_SNDMORE);

        zmq::message_t partitionData(bufSize);
        std::memcpy(partitionData.data(), partitionStart, bufSize);
        workersocket.send(partitionData);
    }
}

void
LambdaWorker::recvBackpropFinishMsg(zmq::message_t& client_id) {

    // Send confirm ACK message.
    zmq::message_t confirm;
    workersocket.send(client_id, ZMQ_SNDMORE);
    workersocket.send(confirm);

    // Wake up lambdaComm.
    std::lock_guard<std::mutex> lk(finish_mutex);
    finishedBackward = true;
    cv_backward.notify_one();
}


///////////////////////////////////
// Below are LambdaComm methods. //
///////////////////////////////////


/**
 *
 * Call 'newContext()' before the lambda invokation to refresh the parameters, then call `requestLambdas()` to tell the coordserver to
 * trigger lambda threads.
 * 
 */
void
LambdaComm::newContextForward(FeatType *dataBuf, FeatType *zData, FeatType *actData,
                              unsigned numLocalVertices, unsigned numFeats, unsigned numFeatsNext) {
    countForward = 0;

    // Create a new matrix object for workers to access.
    Matrix actMatrix(numLocalVertices, numFeats, dataBuf);

    // Refresh workers' members, and connect their worker sockets to the backend.
    for (auto&& worker : workers)
        worker->refreshState(actMatrix, zData, actData, numFeatsNext);

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
    std::unique_lock<std::mutex> lk(count_mutex);
    cv_forward.wait(lk, [&]{ return countForward == numLambdasForward; });
}


/**
 *
 * Call 'newContext()' before the lambda invokation to refresh the parameters, then call `requestLambdas()` to tell the coordserver to
 * trigger lambda threads.
 * 
 */
void
LambdaComm::newContextBackward(std::vector<FeatType *> zBufs, std::vector<FeatType *> actBufs, FeatType *targetBuf,
                               unsigned numLocalVertices, std::vector<unsigned> layerConfig) {
    finishedBackward = false;

    // Create new matrix objects for workers to access.
    std::vector<Matrix> zMatrices;
    assert(zBufs.size() == layerConfig.size());
    for (size_t i = 1; i < layerConfig.size(); ++i)
        zMatrices.push_back(Matrix(numLocalVertices, layerConfig[i], zBufs[i]));

    std::vector<Matrix> actMatrices;
    assert(actBufs.size() == layerConfig.size());
    for (size_t i = 0; i < layerConfig.size(); ++i)
        actMatrices.push_back(Matrix(numLocalVertices, layerConfig[i], actBufs[i]));

    Matrix targetMatrix(numLocalVertices, layerConfig[layerConfig.size() - 1], targetBuf);

    // Refresh workers' members.
    for (auto&& worker : workers)
        worker->refreshState(zMatrices, actMatrices, targetMatrix);

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
    std::unique_lock<std::mutex> lk(finish_mutex);
    cv_backward.wait(lk, [&]{ return finishedBackward; });
}


/**
 * 
 * Send message to the coordination server to shutdown.
 * 
 */
void
LambdaComm::sendShutdownMessage() {

    // Send kill message.
    zmq::message_t header(HEADER_SIZE);
    populateHeader((char *) header.data(), OP::TERM);
    coordsocket.send(header, ZMQ_SNDMORE);

    // Send dummy message since coordination server expects an IP as well.
    zmq::message_t dummyIP;
    coordsocket.send(dummyIP);
}
