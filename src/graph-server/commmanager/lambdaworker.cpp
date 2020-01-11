#include "lambdaworker.hpp"
#include "lambda_comm.hpp"



/**
 *
 * LambdaWorker constructor & destructor.
 *
 */
LambdaWorker::LambdaWorker(LambdaComm *manager_) : manager(manager_), workersocket(manager->ctx, ZMQ_DEALER) {
    workersocket.setsockopt(ZMQ_LINGER, 0);
    workersocket.setsockopt(ZMQ_RCVTIMEO, 1000); // Set time out of weight socket to 1s for a graceful shut down.
    workersocket.connect("inproc://backend");
}

LambdaWorker::~LambdaWorker() {
    workersocket.close();
}


/**
 *
 * Lambdaworker is a wrapper over the sender & receiver thread.
 *
 */
void
LambdaWorker::work() {
    try {
        while (!(manager->halt)) {
            zmq::message_t identity;
            zmq::message_t header;

            // recv will return false if timed out.
            if (!workersocket.recv(&identity)) {
                continue;
            }
            if (identity.size() != sizeof(unsigned) * 3 + manager->nodeIp.size()) {
                printLog(manager->nodeId, "identity size %u", identity.size());
                continue;
            }
            if (!workersocket.recv(&header)) {
                continue;
            }
            if (header.size() != HEADER_SIZE) {
                printLog(manager->nodeId, "header size %u", header.size());
                continue;
            }

            unsigned op = parse<unsigned>((char *) header.data(), 0);
            unsigned partId2 = parse<unsigned>((char *) header.data(), 1);
            unsigned partId = parse<unsigned>((char *) identity.data(), 0);
            if (partId != partId2) {
                printLog(manager->nodeId, "partIds don't match! op %u, partId: %u, %u; %u %u", op, partId, partId2, identity.size(), header.size());
            }
            unsigned layer = parse<unsigned>((char *) identity.data(), 1);
            if (layer != manager->currLayer) {
                printLog(manager->nodeId, "layer %u %u", layer, manager->currLayer);
                workersocket.send(identity, ZMQ_SNDMORE);
                zmq::message_t header(HEADER_SIZE);
                populateHeader((char *) header.data(), -2, -2, -2, -2);
                workersocket.send(header);
                printLog(manager->nodeId, "Discard an old lambda execution");

                // fake recv chunk
                if (op == OP::PUSH_FORWARD) {
                    zmq::message_t data;
                    workersocket.recv(&data);
                    workersocket.recv(&data);
                } else if (op == OP::PUSH_BACKWARD) {
                    zmq::message_t data;
                    workersocket.recv(&data);
                }
                continue;
            }

            switch (op) {
                case (OP::PULL_FORWARD): {
                    if (partId >= manager->numLambdasForward) {
                        printLog(manager->nodeId, "error partid %u!", partId);
                        break;
                    }
                    if (manager->forwardLambdaTable[partId]) {
                        // printLog(manager->nodeId, "recv req from %d", partId);
                        sendAggregatedChunk(identity, partId);
                        // printLog(manager->nodeId, "send chunk to %d", partId);
                    } else {
                        workersocket.send(identity, ZMQ_SNDMORE);
                        zmq::message_t header(HEADER_SIZE);
                        populateHeader((char *) header.data(), -2, -2, -2, -2);
                        workersocket.send(header);
                        printLog(manager->nodeId, "Discard old lambda %u execution", partId);
                    }
                    break;
                }
                case (OP::PUSH_FORWARD): {
                    if (partId >= manager->numLambdasForward) {
                        printLog(manager->nodeId, "error partid %u!", partId);
                        break;
                    }
                    if (manager->forwardLambdaTable[partId]) {
                        recvLambdaResults(identity, partId);
                    } else {
                        fakeRecvChunks(identity, 2);
                    }
                    break;
                }
                case (OP::PULL_BACKWARD): {
                    if (partId >= manager->numLambdasBackward) {
                        printLog(manager->nodeId, "error partid %u!", partId);
                        break;
                    }
                    if (manager->backwardLambdaTable[partId]) {
                        // printLog(manager->nodeId, "recv req from %d", partId);
                        unsigned matType = parse<unsigned>((char *) header.data(), 2);
                        if (matType == TYPE::GRAD) {
                            sendChunk(oldGradMatrix, identity, partId, false);
                        } else if (matType == TYPE::AH || matType == TYPE::Z || matType == TYPE::ACT) {
                            unsigned layer = parse<unsigned>((char *) header.data(), 3);
                            sendChunk(savedTensors[layer][matType - 1], identity, partId, false);
                        } else if (matType == TYPE::LAB) {
                            sendChunk(targetMatrix, identity, partId, false);
                        }
                        // printLog(manager->nodeId, "send chunk to %d", partId);
                    } else {
                        workersocket.send(identity, ZMQ_SNDMORE);
                        zmq::message_t header(HEADER_SIZE);
                        populateHeader((char *) header.data(), -2, -2, -2, -2);
                        workersocket.send(header);
                        printLog(manager->nodeId, "Discard old lambda %u execution", partId);
                    }
                    break;
                }
                case (OP::PULL_EVAL):
                    if (manager->forwardLambdaTable[partId]) {
                        sendTargetMatrix(identity, partId);
                    }
                    break;
                case (OP::PUSH_EVAL):
                    break;
                case (OP::PUSH_BACKWARD): {
                    if (partId >= manager->numLambdasBackward) {
                        printLog(manager->nodeId, "error partid %u!", partId);
                        break;
                    }
                    if (manager->backwardLambdaTable[partId]) {
                        recvChunk(newGradMatrix, identity, partId, false);
                    } else {
                        fakeRecvChunks(identity, 1);
                    }
                    break;
                }
                default:
                    {
                        printLog(manager->nodeId, "unknown op %d, part id %d", op, partId);
                    }
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
LambdaWorker::refreshState(Matrix actMatrix_, FeatType *zData_, FeatType *actData_, unsigned numFeatsNext_) { // For forward-prop.
    actMatrix = actMatrix_;
    zData = zData_;
    actData = actData_;
    numFeatsNext = numFeatsNext_;
}

void
LambdaWorker::refreshState(Matrix oldGradMatrix_, Matrix newGradMatrix_, Matrix targetMatrix_, std::vector<Matrix> *savedTensors_) { // For backward-prop.
    oldGradMatrix = oldGradMatrix_;
    newGradMatrix = newGradMatrix_;
    targetMatrix = targetMatrix_;
    savedTensors = savedTensors_;
}


/**
 *
 * Sending & receiving messages to / from lambda threads.
 *
 */
void
LambdaWorker::sendAggregatedChunk(zmq::message_t& client_id, unsigned partId) {
    // Reject a send request if the partition id is invalid.
    if (partId >= manager->numLambdasForward) {
        workersocket.send(client_id, ZMQ_SNDMORE);
        zmq::message_t header(HEADER_SIZE);
        populateHeader((char *) header.data(), ERR_HEADER_FIELD, ERR_HEADER_FIELD, ERR_HEADER_FIELD, ERR_HEADER_FIELD);
        workersocket.send(header);

        printLog(manager->nodeId, "[ERROR] Got a request for partition %u, but number of lambdas is %u", partId, manager->numLambdasForward);

    // Partition id is valid, so send the matrix segment.
    } else {
        workersocket.send(client_id, ZMQ_SNDMORE);

        // Check to make sure that the bounds of this partition do not exceed the bounds of the data array.
        // If they do, set partition end to the end of the array.
        unsigned partRows = std::ceil((float) actMatrix.getRows() / (float) manager->numLambdasForward);
        unsigned thisPartRows = partRows;
        if ((partId * partRows + partRows) > actMatrix.getRows())
            thisPartRows = partRows - (partId * partRows + partRows) + actMatrix.getRows();
        unsigned bufSize = thisPartRows * actMatrix.getCols() * sizeof(FeatType);
        FeatType *partitionStart = actMatrix.getData() + (partId * partRows * actMatrix.getCols());

        zmq::message_t header(HEADER_SIZE);
        populateHeader((char *) header.data(), OP::RESP, 0, thisPartRows, actMatrix.getCols());
        workersocket.send(header, ZMQ_SNDMORE);

        zmq::message_t partitionData(bufSize);
        std::memcpy(partitionData.data(), partitionStart, bufSize);
        workersocket.send(partitionData);
    }
}

void
LambdaWorker::recvLambdaResults(zmq::message_t& client_id, unsigned partId) {
    unsigned partRows = std::ceil((float) actMatrix.getRows() / (float) (manager->numLambdasForward));
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
    // manager->forwardLambdaTable[partId] = false;
    // ++(manager->countForward);
    __sync_bool_compare_and_swap(manager->forwardLambdaTable + partId, true, false);
    __sync_fetch_and_add(&(manager->countForward), 1);
}

void
LambdaWorker::fakeRecvChunks(zmq::message_t& client_id, unsigned chunkCnt) {
    zmq::message_t data;
    for (unsigned i = 0; i < chunkCnt; ++i) {
        workersocket.recv(&data);
    }

    // Send confirm ACK message.
    zmq::message_t confirm;
    workersocket.send(client_id, ZMQ_SNDMORE);
    workersocket.send(confirm);
}

void
LambdaWorker::sendChunk(Matrix &srcMat, zmq::message_t& client_id, unsigned partId, bool forward) {
    // Reject a send request if the partition id is invalid.
    unsigned numLambdas = forward ? (manager->numLambdasForward) : (manager->numLambdasBackward);
    if (partId >= numLambdas) {
        workersocket.send(client_id, ZMQ_SNDMORE);
        zmq::message_t header(HEADER_SIZE);
        populateHeader((char *) header.data(), ERR_HEADER_FIELD, ERR_HEADER_FIELD, ERR_HEADER_FIELD, ERR_HEADER_FIELD);
        workersocket.send(header);

        printLog(manager->nodeId, "[ERROR] Got a request for partition %u, but number of lambdas is %u", partId, numLambdas);
    // Partition id is valid, so send the matrix segment.
    } else {
        workersocket.send(client_id, ZMQ_SNDMORE);

        // Check to make sure that the bounds of this partition do not exceed the bounds of the data array.
        // If they do, set partition end to the end of the array.
        unsigned partRows = (srcMat.getRows() + numLambdas - 1) / numLambdas;
        unsigned thisPartRows = std::min(partRows, srcMat.getRows() - partId * partRows);
        unsigned bufSize = thisPartRows * srcMat.getCols() * sizeof(FeatType);
        FeatType *partitionStart = srcMat.getData() + (partId * partRows * srcMat.getCols());

        zmq::message_t header(HEADER_SIZE);
        populateHeader((char *) header.data(), OP::RESP, 0, thisPartRows, srcMat.getCols());
        workersocket.send(header, ZMQ_SNDMORE);

        zmq::message_t partitionData(bufSize);
        std::memcpy(partitionData.data(), partitionStart, bufSize);
        workersocket.send(partitionData);
    }
}

void
LambdaWorker::recvChunk(Matrix &dstMat, zmq::message_t &client_id, unsigned partId, bool forward) {
    unsigned numLambdas = forward ? (manager->numLambdasForward) : (manager->numLambdasBackward);
    unsigned partRows = (dstMat.getRows() + numLambdas - 1) / numLambdas;
    FeatType *partitionStart = dstMat.getData() + partId * partRows * dstMat.getCols();


    // Receive the pushed-back results.
    zmq::message_t msg;
    workersocket.recv(&msg);
    std::memcpy(partitionStart, msg.data(), msg.size());

    // Send confirm ACK message.
    zmq::message_t confirm;
    workersocket.send(client_id, ZMQ_SNDMORE);
    workersocket.send(confirm);

    // Check for total number of partitions received. If all partitions received, wake up lambdaComm.
    if (forward) {
        // manager->forwardLambdaTable[partId] = false;
        // ++(manager->countForward);
        __sync_bool_compare_and_swap(manager->forwardLambdaTable + partId, true, false);
        __sync_fetch_and_add(&(manager->countForward), 1);
    }
    else{
        // manager->backwardLambdaTable[partId] = false;
        // ++(manager->countBackward);
        __sync_bool_compare_and_swap(manager->backwardLambdaTable + partId, true, false);
        __sync_fetch_and_add(&(manager->countBackward), 1);
    }
}


void
LambdaWorker::sendTargetMatrix(zmq::message_t& client_id, unsigned partId) {
    unsigned partRows = std::ceil((float) targetMatrix.getRows() / (float) (manager->numLambdasForward));
    unsigned thisPartRows = partRows;
    if ((partId * partRows + partRows) > targetMatrix.getRows())
        thisPartRows = partRows - (partId * partRows + partRows) + targetMatrix.getRows();

    unsigned bufSize = thisPartRows * targetMatrix.getCols() * sizeof(FeatType);
    FeatType* partitionStart = targetMatrix.getData() + (partId * partRows * targetMatrix.getCols());

    zmq::message_t header(HEADER_SIZE);
    populateHeader((char*) header.data(), OP::RESP, 0, thisPartRows, targetMatrix.getCols());

    workersocket.send(client_id, ZMQ_SNDMORE);
    workersocket.send(header, ZMQ_SNDMORE);

    zmq::message_t partitionData(bufSize);
    std::memcpy(partitionData.data(), partitionStart, bufSize);
    workersocket.send(partitionData);
}
