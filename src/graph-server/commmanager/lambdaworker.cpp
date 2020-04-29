#include "lambdaworker.hpp"
#include "lambda_comm.hpp"

#include <chrono>
#include <iomanip>

static void nofree(void* data, void* hint) {}

/**
 *
 * LambdaWorker constructor & destructor.
 *
 */
LambdaWorker::LambdaWorker(LambdaComm *manager_) :
  manager(manager_), workersocket(manager->ctx, ZMQ_DEALER) {
    workersocket.setsockopt(ZMQ_RCVTIMEO, 1000); // Set time out of weight socket to 1s for a graceful shut down.
    workersocket.connect("inproc://backend");
}

LambdaWorker::~LambdaWorker() {
    workersocket.setsockopt(ZMQ_LINGER, 0);
    workersocket.close();
}


/**
 *
 * Lambdaworker is a wrapper over the sender & receiver thread.
 *
 */
void
LambdaWorker::work(unsigned _wid) {
    wid = _wid;
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
            Chunk chunk;
            memcpy(&chunk, (char *)header.data() + sizeof(OP), sizeof(Chunk));

            switch (op) {
                case (OP::PULL): {
                    sendTensors(identity, chunk);
                    break;
                }
                case (OP::PUSH): {
                    recvTensors(identity, chunk);
                    break;
                }
                case (OP::EVAL): {
                    recvEvalData(identity, chunk);
                    break;
                }
                case (OP::FIN): {
                    markFinish(identity, chunk);
                    break;
                }
                default: {
                    printLog(manager->nodeId, "unknown op %d, part id %d", op, chunk.localId);
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
void LambdaWorker::sendTensors(zmq::message_t& client_id, Chunk &chunk) {
    manager->timeoutMtx.lock();
    bool exist = manager->timeoutTable.find(chunk) != manager->timeoutTable.end();
    manager->timeoutMtx.unlock();
    if (exist) {
        workersocket.send(client_id, ZMQ_SNDMORE);

        TensorMap& tensorMap = manager->savedNNTensors[chunk.layer];
        unsigned more = 1;
        while (more) {
            zmq::message_t tensorHeader(TENSOR_HDR_SIZE);
            workersocket.recv(&tensorHeader);

            std::string name = parseName((char*)tensorHeader.data());
            auto found = tensorMap.find(name);
            if (found == tensorMap.end()) {
                printLog(manager->nodeId, "Requested tensor '%s' not found", name.c_str());
                zmq::message_t errorHeader(TENSOR_HDR_SIZE);
                populateHeader(errorHeader.data(), ERR_HEADER_FIELD, name.c_str());
                workersocket.send(client_id, ZMQ_SNDMORE);
                workersocket.send(errorHeader);
                return;
            } else {
                Matrix& reqMatrix = found->second;
                sendTensor(reqMatrix, chunk, more);
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
        workersocket.send(header);

        char errMsg[1024];
        sprintf(errMsg, "[ ERROR ] when sending chunk: %s %u",
            chunk.str().c_str(), chunk.vertex);
        printLog(manager->nodeId, errMsg);
    }
}

void LambdaWorker::recvTensors(zmq::message_t& client_id, Chunk &chunk) {
    manager->timeoutMtx.lock();
    bool exist = manager->timeoutTable.find(chunk) != manager->timeoutTable.end();
    manager->timeoutMtx.unlock();
    if (exist) {
        int ret = 0;
        unsigned more = 1;
        size_t usize = sizeof(unsigned);
        while (more && ret == 0) {
            ret = recvTensor(chunk);
            workersocket.getsockopt(ZMQ_RCVMORE, &more, &usize);
        }

        if (ret == 0 && manager->NNRecv(chunk)) {
            zmq::message_t ack;
            workersocket.send(client_id, ZMQ_SNDMORE);
            workersocket.send(ack);
        } else { // Error, Give up this chunk
            zmq::message_t ack(3 * sizeof(unsigned));
            *(int *)(ack.data()) = -1;
            workersocket.send(client_id, ZMQ_SNDMORE);
            workersocket.send(ack);
        }
    } else {
        zmq::message_t tensorHeader(TENSOR_HDR_SIZE);
        unsigned more = 1;
        size_t usize = sizeof(unsigned);
        while (more) {
            zmq::message_t tensorData;
            workersocket.recv(&tensorHeader);
            workersocket.recv(&tensorData);

            workersocket.getsockopt(ZMQ_RCVMORE, &more, &usize);
        }
        zmq::message_t ack(3 * sizeof(unsigned));
        *(int *)(ack.data()) = -1;
        workersocket.send(client_id, ZMQ_SNDMORE);
        workersocket.send(ack);

        std::string errMsg = "[ ERROR ] when receiving from " + chunk.str() + ": ";
        errMsg += "Received duplicate results. Discarding...";
        printLog(manager->nodeId, errMsg.c_str());
    }
}

void LambdaWorker::recvEvalData(zmq::message_t &client_id, Chunk &chunk) {
    manager->timeoutMtx.lock();
    bool exist = manager->timeoutTable.find(chunk) != manager->timeoutTable.end();
    manager->timeoutMtx.unlock();
    if (exist) {
        zmq::message_t evalMsg(2 * sizeof(float));
        workersocket.recv(&evalMsg);

        float acc = *((float *)evalMsg.data());
        float loss = *(((float *)evalMsg.data()) + 1);

        manager->accMtx.lock();
        auto &accLoss = manager->accLossTable[chunk.epoch];
        accLoss.acc += acc;
        accLoss.loss += loss;
        accLoss.vtcsCnt += chunk.upBound - chunk.lowBound;
        accLoss.chunkCnt++;
        // printLog(manager->nodeId, "epoch %u, chunk %u/%u, acc %.3f, loss %.3f", chunk.epoch, accLoss.chunkCnt, manager->engine->numLambdasForward,
        //                 accLoss.acc / accLoss.vtcsCnt, accLoss.loss / accLoss.vtcsCnt);
        if (accLoss.chunkCnt == manager->engine->numLambdasForward) {
            printLog(manager->nodeId, "%u: epoch %u, acc %.3f, loss %.3f", timestamp_ms(), chunk.epoch, accLoss.acc / accLoss.vtcsCnt, accLoss.loss / accLoss.vtcsCnt);
        }
        manager->accMtx.unlock();
    } else {
        std::string errMsg = "[ ERROR ] when receiving from " + chunk.str() + ": ";
        errMsg += "Received duplicate results. Discarding...";
        printLog(manager->nodeId, errMsg.c_str());
    }
}

void LambdaWorker::markFinish(zmq::message_t& client_id, Chunk &chunk) {
    manager->timeoutMtx.lock();
    bool exist = manager->timeoutTable.find(chunk) != manager->timeoutTable.end();
    manager->timeoutMtx.unlock();
    if (exist) {
        zmq::message_t ack(3 * sizeof(unsigned));
        if (manager->async && manager->enqueueAggChunk(chunk)) {
            *(int *)(ack.data()) = 0;
        } else if (manager->NNRecv(chunk)) {
            *(int *)(ack.data()) = 0;
        } else { // Error, Give up this chunk
            *(int *)(ack.data()) = -1;
        }
        workersocket.send(client_id, ZMQ_SNDMORE);
        workersocket.send(ack);
    } else {
        zmq::message_t ack(3 * sizeof(unsigned));
        *(int *)(ack.data()) = -1;
        workersocket.send(client_id, ZMQ_SNDMORE);
        workersocket.send(ack);

        std::string errMsg = "[ ERROR ] when receiving from " + chunk.str() + ": ";
        errMsg += "Received duplicate results. Discarding...";
        printLog(manager->nodeId, errMsg.c_str());
    }
}

void LambdaWorker::sendTensor(Matrix &tensor, Chunk &chunk, unsigned& more) {
    FeatType *dptr = tensor.get(chunk.lowBound);
    unsigned rows = chunk.upBound - chunk.lowBound;
    unsigned cols = tensor.getCols();

    zmq::message_t responseHeader(TENSOR_HDR_SIZE);
    populateHeader(responseHeader.data(), OP::PULL, tensor.name().c_str(),
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

int LambdaWorker::recvTensor(Chunk &chunk) {
    zmq::message_t tensorHeader(TENSOR_HDR_SIZE);
    workersocket.recv(&tensorHeader);
    zmq::message_t tensorData;
    workersocket.recv(&tensorData);

    std::string name = parseName((char*)tensorHeader.data());
    TensorMap& tensorMap = manager->savedNNTensors[chunk.layer];
    auto found = tensorMap.find(name);
    if (found == tensorMap.end()) {
        printLog(manager->nodeId, "Lambda %s returned unknown tensor '%s'. Make sure to allocate it before running lambdas!",
                 chunk.str().c_str(), name.c_str());
        return 1;
     }

    FeatType* dptr = found->second.get(chunk.lowBound);
    std::memcpy(dptr, tensorData.data(), tensorData.size());

    return 0;
}

// TODO: (YIFAN) This is outdated.
void LambdaWorker::sendRefChunk(Matrix &srcMat, zmq::message_t& client_id, unsigned partId, bool forward) {
    // // Reject a send request if the partition id is invalid.
    // unsigned numLambdas = manager->numLambdasForward;
    //     workersocket.send(client_id, ZMQ_SNDMORE);
    //     zmq::message_t header(HEADER_SIZE);
    //     populateHeader((char *) header.data(), ERR_HEADER_FIELD, ERR_HEADER_FIELD, ERR_HEADER_FIELD, ERR_HEADER_FIELD);
    //     workersocket.send(header);

    //     printLog(manager->nodeId, "[ERROR] Got a request for partition %u, but number of lambdas is %u", partId, numLambdas);
    // // Partition id is valid, so send the matrix segment.
    // } else {
    //     workersocket.send(client_id, ZMQ_SNDMORE);

    //     // Check to make sure that the bounds of this partition do not exceed the bounds of the data array.
    //     // If they do, set partition end to the end of the array.
    //     unsigned partRows = (srcMat.getRows() + numLambdas - 1) / numLambdas;
    //     unsigned thisPartRows = std::min(partRows, srcMat.getRows() - partId * partRows);
    //     unsigned featDim = srcMat.getCols();
    //     FeatType **chunk = (FeatType **)srcMat.getData() + partId * partRows;

    //     zmq::message_t header(HEADER_SIZE);
    //     populateHeader((char *) header.data(), OP::RESP, 0, thisPartRows, featDim);
    //     workersocket.send(header, ZMQ_SNDMORE);

    //     zmq::message_t chunkMsg(thisPartRows * featDim * sizeof(FeatType));
    //     for (unsigned i = 0; i < thisPartRows; ++i) {
    //         memcpy((FeatType *)chunkMsg.data() + i * featDim, chunk[i], sizeof(FeatType) * featDim);
    //     }

    //     workersocket.send(chunkMsg);
    // }
}
