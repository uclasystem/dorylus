#include "lambdaworker.hpp"
#include "lambda_comm.hpp"

#include <chrono>
#include <iomanip>

static void nofree(void* data, void* hint) {}

extern std::mutex producerQueueLock;

/**
 *
 * LambdaWorker constructor & destructor.
 *
 */
LambdaWorker::LambdaWorker(LambdaComm *manager_) :
  manager(manager_), workersocket(manager->ctx, ZMQ_DEALER) {
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
            recvTS = timestamp_ms();

            OP op = parse<OP>((char *) header.data(), 0);
            unsigned partId2 = parse<unsigned>((char *) header.data(), 1);
            unsigned partId = parse<unsigned>((char *) identity.data(), 0);
            if (partId != partId2) {
                printLog(manager->nodeId, "partIds don't match! op %u, partId: %u, %u; %u %u", op, partId, partId2, identity.size(), header.size());
            }

            unsigned layer = parse<unsigned>((char *) identity.data(), 1);

            switch (op) {
                case (OP::PULL_VTX_FORWARD): {
                    sendChunk(manager->inputTensor, identity, partId, layer, true);
                    break;
                }
                case (OP::PUSH_VTX_FORWARD): {
                    recvLambdaResults(identity, partId, layer);
                    break;
                }
                case (OP::PULL_VTX_BACKWARD): {
                    sendGCNChunks(identity, partId, layer);
                    break;
                }
                case (OP::PUSH_VTX_BACKWARD): {
                    recvChunk(manager->outputTensor, identity, partId, layer, false);
                    break;
                }
                case (OP::PULL_VTX_EVAL): {
                    sendTargetMatrix(identity, partId);
                    break;
                }
                case (OP::PUSH_VTX_EVAL):
                case (OP::PULL_EDG_EVAL):
                case (OP::PUSH_EDG_EVAL):
                    break;
                case (OP::PULL): {
                    sendTensors(partId, layer, identity);
                    break;
                }
                case (OP::PUSH): {
                    recvTensors(partId, layer, identity);
                    break;
                }
                case (OP::PULL_EDG_FORWARD):
                case (OP::PULL_EDG_BACKWARD):
                case (OP::PUSH_EDG_FORWARD):
                case (OP::PUSH_EDG_BACKWARD):
                default: {
                    printLog(manager->nodeId, "unknown op %d, part id %d", op, partId);
                    break;  /** Not an op that I care about. */
                }
            }
        }
    } catch (std::exception& ex) { /** Context Termintated. */ }
}


/**
 *
 * Sending & receiving messages to / from lambda threads.
 *
 */
void
LambdaWorker::sendChunk(Matrix &srcMat, zmq::message_t& client_id, unsigned partId, unsigned layer, bool forward) {
    unsigned numLambdas = forward ? (manager->numLambdasForward) : (manager->numLambdasBackward);
    // Partition id is valid, so send the matrix segment.
    if (manager->forward && layer == manager->currLayer && partId < numLambdas &&
        manager->forwardLambdaTable[partId]) {
        // Check to make sure that the bounds of this partition do not exceed the bounds of the data array.
        // If they do, set partition end to the end of the array.
        unsigned partRows = (srcMat.getRows() + numLambdas - 1) / numLambdas;
        unsigned thisPartRows = std::min(partRows, srcMat.getRows() - partId * partRows);
        unsigned bufSize = thisPartRows * srcMat.getCols() * sizeof(FeatType);
        FeatType *chunk = srcMat.get(partId * partRows);

        zmq::message_t header(HEADER_SIZE);
        populateHeader((char *) header.data(), OP::RESP, 0, thisPartRows, srcMat.getCols());
        setTSinHdr(header.data());

        zmq::message_t partitionData(bufSize);
        std::memcpy(partitionData.data(), chunk, bufSize);

        workersocket.send(client_id, ZMQ_SNDMORE);
        workersocket.send(header, ZMQ_SNDMORE);
        workersocket.send(partitionData);
    // Reject a send request if the partition id is invalid.
    } else {
        workersocket.send(client_id, ZMQ_SNDMORE);
        zmq::message_t header(HEADER_SIZE);
        populateHeader((char *) header.data(), ERR_HEADER_FIELD, ERR_HEADER_FIELD, ERR_HEADER_FIELD, ERR_HEADER_FIELD);
        setTSinHdr(header.data());
        workersocket.send(header);

        std::string errMsg = "[ ERROR ] (when sending to " + std::to_string(partId) + "): ";
        if (!manager->forward) {
            errMsg += "Forward lambda during backward phase. ";
        }
        if (layer != manager->currLayer) {
            errMsg += std::string("Lambda layer ") + std::to_string(layer) +
                      " doesn't match current layer " + std::to_string(manager->currLayer) + ". ";
        }
        if (partId >= manager->numLambdasForward) {
            errMsg += std::string("Lambda partId ") + std::to_string (partId) + " exceeds numLambdas" + std::to_string(manager->numLambdasForward) + ". ";
        }
        if (!manager->forwardLambdaTable[partId]) {
            errMsg += std::string("Lambda ") + std::to_string(partId) + " has already done. ";
        }
        errMsg += "Discard by graph server.";
        printLog(manager->nodeId, errMsg.c_str());
    }
}

// TODO: (YIFAN) This is outdated.
void
LambdaWorker::sendRefChunk(Matrix &srcMat, zmq::message_t& client_id, unsigned partId, bool forward) {
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
        unsigned featDim = srcMat.getCols();
        FeatType **chunk = (FeatType **)srcMat.getData() + partId * partRows;

        zmq::message_t header(HEADER_SIZE);
        populateHeader((char *) header.data(), OP::RESP, 0, thisPartRows, featDim);
        workersocket.send(header, ZMQ_SNDMORE);

        zmq::message_t chunkMsg(thisPartRows * featDim * sizeof(FeatType));
        for (unsigned i = 0; i < thisPartRows; ++i) {
            memcpy((FeatType *)chunkMsg.data() + i * featDim, chunk[i], sizeof(FeatType) * featDim);
        }

        workersocket.send(chunkMsg);
    }
}

// TODO: (YIFAN) there is no reason to receive all data together. We should receive one chunk per time.
void
LambdaWorker::recvLambdaResults(zmq::message_t& client_id, unsigned partId, unsigned layer) {
    // Partition id is valid, so send the matrix segment.
    if (manager->forward && layer == manager->currLayer && partId < manager->numLambdasForward &&
        manager->forwardLambdaTable[partId]) {

        unsigned partRows = std::ceil((float) manager->inputTensor.getRows() / (float) (manager->numLambdasForward));
        // TODO: (YIFAN) generate this. For GCN there is only 1 saved tensor: the z tensor
        FeatType *savedTensorChunk = manager->savedTensors[layer][TYPE::Z - 1].get(partId * partRows);
        FeatType *outputChunk = manager->outputTensor.get(partId * partRows);

        // Receive the pushed-back results.
        zmq::message_t data;
        workersocket.recv(&data);
        std::memcpy(savedTensorChunk, data.data(), data.size());
        workersocket.recv(&data);
        std::memcpy(outputChunk, data.data(), data.size());

        // Send confirm ACK message.
        zmq::message_t confirm(3 * sizeof(unsigned));
        *(int *)(confirm.data()) = 0; // 0 : recving success
        setTSinCfm(confirm.data());
        workersocket.send(client_id, ZMQ_SNDMORE);
        workersocket.send(confirm);

        if (manager->pipeline) {
            producerQueueLock.lock();
            if (manager->forwardLambdaTable[partId]) {
                manager->queuePtr->push(std::make_pair(partId, partRows));
            }
            producerQueueLock.unlock();
        }

        // Check for total number of partitions received. If all partitions received, wake up lambdaComm.
        __sync_bool_compare_and_swap(manager->forwardLambdaTable + partId, true, false);
        __sync_fetch_and_add(&(manager->countForward), 1);
    // Reject a send request if the request is invalid.
    } else {
        fakeRecvChunks(client_id, 2);

        std::string errMsg = "[ ERROR ] (when recving from " + std::to_string(partId) + "): ";
        if (!manager->forward) {
            errMsg += "Forward lambda during backward phase. ";
        }
        if (layer != manager->currLayer) {
            errMsg += std::string("Lambda layer ") + std::to_string(layer) +
                      " doesn't match current layer " + std::to_string(manager->currLayer) + ". ";
        }
        if (partId >= manager->numLambdasForward) {
            errMsg += std::string("Lambda partId ") + std::to_string (partId) + " exceeds numLambdas" + std::to_string(manager->numLambdasForward) + ". ";
        }
        if (!manager->forwardLambdaTable[partId]) {
            errMsg += std::string("Lambda ") + std::to_string(partId) + " has already done. ";
        }
        errMsg += "Discard by graph server.";
        printLog(manager->nodeId, errMsg.c_str());
    }
}

void
LambdaWorker::sendGCNChunks(zmq::message_t& client_id, unsigned partId, unsigned layer) {
    unsigned numLambdas = manager->numLambdasBackward;
    // Partition id is valid, so send the matrix segment.
    if (!manager->forward && layer == manager->currLayer && partId < numLambdas &&
        manager->backwardLambdaTable[partId]) {
        // Check to make sure that the bounds of this partition do not exceed the bounds of the data array.
        // If they do, set partition end to the end of the array.
        const unsigned partRows = (manager->inputTensor.getRows() + numLambdas - 1) / numLambdas;
        const unsigned thisPartRows = std::min(partRows, manager->inputTensor.getRows() - partId * partRows);
        unsigned bufSize;
        FeatType *partitionStart;

        std::vector<Matrix> mats;
        if (layer == 0) { // gradLayer
            mats.push_back(manager->inputTensor);                        // grad mat
            mats.push_back(manager->savedTensors[layer][TYPE::Z - 1]);   // z mat
            mats.push_back(manager->savedTensors[layer][TYPE::AH - 1]);  // ah mat
        } else if (layer == 1) { // gradLoss
            mats.push_back(manager->savedTensors[layer][TYPE::ACT - 1]); // act mat
            mats.push_back(manager->targetTensor);                       // lab mat
            mats.push_back(manager->savedTensors[layer][TYPE::AH - 1]);  // ah mat
        }
        if (mats.size() > 0) {
            workersocket.send(client_id, ZMQ_SNDMORE);
                // prepare header and send
            zmq::message_t header(HEADER_SIZE);
            populateHeader((char *) header.data(), OP::RESP, 0, thisPartRows, mats[0].getCols());
            setTSinHdr(header.data());
            workersocket.send(header, ZMQ_SNDMORE);
            for (unsigned i = 0; i < mats.size() - 1; ++i) {
                bufSize = thisPartRows * mats[i].getCols() * sizeof(FeatType);
                partitionStart = mats[i].getData() + (partId * partRows * mats[i].getCols());
                zmq::message_t matMsg(bufSize);
                memcpy(matMsg.data(), partitionStart, bufSize);
                workersocket.send(matMsg, ZMQ_SNDMORE);
            }
            bufSize = thisPartRows * mats[mats.size() - 1].getCols() * sizeof(FeatType);
            partitionStart = mats[mats.size() - 1].getData() + (partId * partRows * mats[mats.size() - 1].getCols());
            zmq::message_t matMsg(bufSize);
            memcpy(matMsg.data(), partitionStart, bufSize);
            workersocket.send(matMsg);
        }
    // Reject a send request if the request is invalid.
    } else {
        workersocket.send(client_id, ZMQ_SNDMORE);
        zmq::message_t header(HEADER_SIZE);
        populateHeader((char *) header.data(), ERR_HEADER_FIELD, ERR_HEADER_FIELD, ERR_HEADER_FIELD, ERR_HEADER_FIELD);
        setTSinHdr(header.data());
        workersocket.send(header);

        std::string errMsg = "[ ERROR ] (when sending to " + std::to_string(partId) + "): ";
        if (manager->forward) {
            errMsg += "Backward lambda during forward phase. ";
        }
        if (layer != manager->currLayer) {
            errMsg += std::string("Lambda layer ") + std::to_string(layer) +
                      " doesn't match current layer " + std::to_string(manager->currLayer) + ". ";
        }
        if (partId >= numLambdas) {
            errMsg += std::string("Lambda partId ") + std::to_string (partId) + " exceeds numLambdas" + std::to_string(numLambdas) + ". ";
        }
        if (!manager->backwardLambdaTable[partId]) {
            errMsg += std::string("Lambda ") + std::to_string(partId) + " has already done. ";
        }
        errMsg += "Discard by graph server.";
        printLog(manager->nodeId, errMsg.c_str());
    }
}


void
LambdaWorker::recvChunk(Matrix &dstMat, zmq::message_t &client_id, unsigned partId, unsigned layer, bool forward) {
    unsigned numLambdas = forward ? (manager->numLambdasForward) : (manager->numLambdasBackward);
    bool *lambdaTable = forward ? (manager->forwardLambdaTable) : (manager->backwardLambdaTable);
    // Partition id is valid, so send the matrix segment.
    if (forward == manager->forward && layer == manager->currLayer && partId < numLambdas &&
        lambdaTable[partId]) {
        unsigned numLambdas = forward ? (manager->numLambdasForward) : (manager->numLambdasBackward);
        unsigned partRows = (dstMat.getRows() + numLambdas - 1) / numLambdas;
        FeatType *partitionStart = dstMat.getData() + partId * partRows * dstMat.getCols();

        // Receive the pushed-back results.
        zmq::message_t msg;
        workersocket.recv(&msg);
        std::memcpy(partitionStart, msg.data(), msg.size());

        // Send confirm ACK message.
        zmq::message_t confirm(3 * sizeof(unsigned));
        *(int *)(confirm.data()) = 0;
        setTSinCfm(confirm.data());
        workersocket.send(client_id, ZMQ_SNDMORE);
        workersocket.send(confirm);

        producerQueueLock.lock();
        if (manager->pipeline && manager->backwardLambdaTable[partId]) {
            manager->queuePtr->push(std::make_pair(partId, partRows));
        }
        producerQueueLock.unlock();

        // Check for total number of partitions received. If all partitions received, wake up lambdaComm.
        if (forward) {
            __sync_bool_compare_and_swap(manager->forwardLambdaTable + partId, true, false);
            __sync_fetch_and_add(&(manager->countForward), 1);
        } else {
            __sync_bool_compare_and_swap(manager->backwardLambdaTable + partId, true, false);
            __sync_fetch_and_add(&(manager->countBackward), 1);
        }
    // Reject a send request if the request is invalid.
    } else {
        fakeRecvChunks(client_id, 1);

        std::string errMsg = "[ ERROR ] (when recv from " + std::to_string(partId) + "): ";
        if (forward != manager->forward) {
            errMsg += std::string(forward ? "Forward" : "Backward") + " lambda during " +
                    std::string(manager->forward? "forward" : "backward") + " phase. ";
        }
        if (layer != manager->currLayer) {
            errMsg += std::string("Lambda layer ") + std::to_string(layer) +
                    " doesn't match current layer " + std::to_string(manager->currLayer) + ". ";
        }
        if (partId >= numLambdas) {
            errMsg += std::string("Lambda partId ") + std::to_string (partId) + " exceeds numLambdas" + std::to_string(numLambdas) + ". ";
        }
        if (!lambdaTable[partId]) {
            errMsg += std::string("Lambda ") + std::to_string(partId) + " has already done. ";
        }
        errMsg += "Discard by graph server.";
        printLog(manager->nodeId, errMsg.c_str());
    }
}


// named-tensors
void LambdaWorker::sendTensor(FeatType* dptr, std::string tensorName, unsigned rows,
  unsigned cols, unsigned& more) {
    zmq::message_t responseHeader(TENSOR_HDR_SIZE);
    populateHeader(responseHeader.data(), OP::PULL, tensorName.c_str(),
      rows, cols);
    unsigned bufSize = rows * cols * sizeof(FeatType);
    zmq::message_t tensorData(dptr, bufSize, nofree, NULL);

    workersocket.send(responseHeader, ZMQ_SNDMORE);

    size_t usize = sizeof(unsigned);
    workersocket.getsockopt(ZMQ_RCVMORE, &more, &usize);
    if (!more) {
        workersocket.send(tensorData);
    } else {
        workersocket.send(tensorData, ZMQ_SNDMORE);
    }
}

void LambdaWorker::getPartitionInfo(Matrix& tensor, unsigned partId, unsigned& more) {
    unsigned partRows = std::ceil((float) tensor.getRows() / (float) manager->numLambdasForward);
    unsigned thisPartRows = partRows;
    if (((partId + 1) * partRows) > tensor.getRows())
        thisPartRows = partRows - ((partId + 1) * partRows) + tensor.getRows();
    FeatType* partStart = tensor.getData() + (partId * partRows * tensor.getCols());

    sendTensor(partStart, tensor.name(), thisPartRows, tensor.getCols(), more);
}

void LambdaWorker::sendTensors(unsigned partId, unsigned layer, zmq::message_t& client_id) {
    if (partId < manager->numLambdasForward
        && manager->forwardLambdaTable[partId]
        && layer == manager->currLayer) {
        unsigned more = 1;
        workersocket.send(client_id, ZMQ_SNDMORE);

        TensorMap& savedNNTensors = (*savedVtxTensors)[layer];
        while (more) {
            zmq::message_t tensorHeader(TENSOR_HDR_SIZE);
            workersocket.recv(&tensorHeader);

            std::string name = parseName((char*)tensorHeader.data());
            auto found = manager->savedVtxTensors->find(name);
            if (found == manager->savedVtxTensors->end()) {
            auto found = savedNNTensors.find(name);
            if (found == savedNNTensors.end()) {
                printLog(manager->nodeId, "Requested tensor '%s' not found", name.c_str());
                zmq::message_t errorHeader(TENSOR_HDR_SIZE);
                populateHeader(errorHeader.data(), ERR_HEADER_FIELD, name.c_str());
                workersocket.send(client_id, ZMQ_SNDMORE);
                workersocket.send(errorHeader);
                return;
            } else {
                Matrix& reqMatrix = found->second;
                getPartitionInfo(reqMatrix, partId, more);
            }
        }
    } else {
        size_t usize = sizeof(unsigned);
        unsigned more = 1;
        while (more) {
            zmq::message_t tensorHeader(TENSOR_HDR_SIZE);
            workersocket.recv(&tensorHeader);

            workersocket.getsockopt(ZMQ_RCVMORE, &more, &usize);
        }

        workersocket.send(client_id, ZMQ_SNDMORE);
        zmq::message_t header(TENSOR_HDR_SIZE);
        populateHeader((char*) header.data(), ERR_HEADER_FIELD, ERR_HEADER_FIELD, ERR_HEADER_FIELD, ERR_HEADER_FIELD);
        setTSinHdr(header.data());
        workersocket.send(header);

        std::string errMsg = "[ ERROR ] when sending to " + std::to_string(partId) + ": ";
        if (partId >= manager->numLambdasForward) {
            errMsg += " exceeds numLambdas";
        }
        if (!(manager->forwardLambdaTable[partId])) {
            errMsg += " already finished";
        }
        if (layer != manager->currLayer) {
            errMsg += " layer's don't match";
        }
        //errMsg += "Received duplicate requests. Discarding...";
        printLog(manager->nodeId, errMsg.c_str());
    }
}

int LambdaWorker::storeTensorPart(unsigned partId, TensorMap& savedNNTensors) {
    zmq::message_t tensorHeader(TENSOR_HDR_SIZE);
    zmq::message_t tensorData;

    workersocket.recv(&tensorHeader);
    workersocket.recv(&tensorData);

    std::string name = parseName((char*)tensorHeader.data());
    auto found = manager->savedVtxTensors->find(name);
    if (found == manager->savedVtxTensors->end()) {
        printLog(manager->nodeId, "Lambda %u returned unknown tensor '%s'. Make sure to allocate it before running lambdas!",
          partId, name.c_str());
        return 1;
    }

    Matrix& result = found->second;
    unsigned partRows = std::ceil((float) result.getRows() / (float) (manager->numLambdasForward));

    FeatType* partPtr = result.getData() + (partId * partRows * result.getCols());
    std::memcpy(partPtr, tensorData.data(), tensorData.size());

    return 0;
}

void LambdaWorker::recvTensors(unsigned partId, unsigned layer, zmq::message_t& client_id) {
    unsigned more = 1;
    size_t usize = sizeof(unsigned);
    if (partId < manager->numLambdasForward
        && manager->forwardLambdaTable[partId]
        && layer == manager->currLayer) {
        int ret = 0;
        TensorMap& savedNNTensors = (*savedVtxTensors)[layer];
        while (more && ret == 0) {
            ret = storeTensorPart(partId, savedNNTensors);

            workersocket.getsockopt(ZMQ_RCVMORE, &more, &usize);
        }

        if (ret == 0) {
            __sync_bool_compare_and_swap(manager->forwardLambdaTable + partId, true, false);
            __sync_fetch_and_add(&(manager->countForward), 1);
        }
    } else {
        zmq::message_t tensorHeader(TENSOR_HDR_SIZE);
        while (more) {
            zmq::message_t tensorData;
            workersocket.recv(&tensorHeader);
            workersocket.recv(&tensorData);

            workersocket.getsockopt(ZMQ_RCVMORE, &more, &usize);
        }

        std::string errMsg = "[ ERROR ] when receiving from " + std::to_string(partId) + ": ";
        errMsg += "Received duplicate results. Discarding...";
        printLog(manager->nodeId, errMsg.c_str());
    }

    zmq::message_t ack;
    workersocket.send(client_id, ZMQ_SNDMORE);
    workersocket.send(ack);
}
// end named-tensors

void
LambdaWorker::sendTargetMatrix(zmq::message_t& client_id, unsigned partId) {
    // Partition id is valid, so send the matrix segment.
    if (partId < manager->numLambdasForward && manager->forwardLambdaTable[partId]) {
        unsigned partRows = std::ceil((float) manager->targetTensor.getRows() / (float) (manager->numLambdasForward));
        unsigned thisPartRows = partRows;
        if ((partId * partRows + partRows) > manager->targetTensor.getRows())
            thisPartRows = partRows - (partId * partRows + partRows) + manager->targetTensor.getRows();

        unsigned bufSize = thisPartRows * manager->targetTensor.getCols() * sizeof(FeatType);
        FeatType* partitionStart = manager->targetTensor.getData() + (partId * partRows * manager->targetTensor.getCols());

        zmq::message_t header(HEADER_SIZE);
        populateHeader((char*) header.data(), OP::RESP, 0, thisPartRows, manager->targetTensor.getCols());
        setTSinHdr(header.data());

        zmq::message_t partitionData(bufSize);
        std::memcpy(partitionData.data(), partitionStart, bufSize);

        workersocket.send(client_id, ZMQ_SNDMORE);
        workersocket.send(header, ZMQ_SNDMORE);
        workersocket.send(partitionData);
    // Reject a send request if the request is invalid.
    } else {
        workersocket.send(client_id, ZMQ_SNDMORE);
        zmq::message_t header(HEADER_SIZE);
        populateHeader((char *) header.data(), ERR_HEADER_FIELD, ERR_HEADER_FIELD, ERR_HEADER_FIELD, ERR_HEADER_FIELD);
        setTSinHdr(header.data());
        workersocket.send(header);

        std::string errMsg = "[ ERROR ] (when sending to " + std::to_string(partId) + "): ";
        if (!manager->forward) {
            errMsg += "Forward lambda during backward phase. ";
        }
        if (partId >= manager->numLambdasForward) {
            errMsg += std::string("Lambda partId ") + std::to_string (partId) + " exceeds numLambdas" + std::to_string(manager->numLambdasForward) + ". ";
        }
        if (!manager->forwardLambdaTable[partId]) {
            errMsg += std::string("Lambda ") + std::to_string(partId) + " has already done. ";
        }
        errMsg += "Discard by graph server.";
        printLog(manager->nodeId, errMsg.c_str());
    }
}

void
LambdaWorker::fakeRecvChunks(zmq::message_t& client_id, unsigned chunkCnt) {
    zmq::message_t data;
    for (unsigned i = 0; i < chunkCnt; ++i) {
        workersocket.recv(&data);
    }

    // Send confirm ACK message.
    zmq::message_t confirm(3 * sizeof(unsigned));
    *(int *)(confirm.data()) = -1;
    setTSinCfm(confirm.data());
    workersocket.send(client_id, ZMQ_SNDMORE);
    workersocket.send(confirm);
}
