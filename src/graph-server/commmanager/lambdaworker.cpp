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
    const unsigned IDENTITY_SIZE = sizeof(Chunk) + sizeof(unsigned);
    wid = _wid;
    try {
        while (!(manager->halt)) {
            zmq::message_t identity;
            zmq::message_t header;

            // recv will return false if timed out.
            if (!workersocket.recv(&identity)) {
                continue;
            }
            if (identity.size() != IDENTITY_SIZE) {
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
            memcpy(&chunk, (char *)identity.data(), sizeof(Chunk));

            switch (op) {
                case (OP::PULL): {
                    sendTensors(identity, chunk);
                    break;
                }
                case (OP::PULLE): {
                    sendEdgeTensor(identity, chunk);
                    break;
                }
                case (OP::PULLEINFO): {
                    sendEdgeInfo(identity, chunk);
                    break;
                }
                case (OP::PUSH): {
                    recvTensors(identity, chunk);
                    break;
                }
                case (OP::PUSHE): {
                    recvETensors(identity, chunk);
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
                case (OP::TERM): {
                    // terminate by weight server
                    CONVERGE_STATE cs = (CONVERGE_STATE)parse<int>((char *) header.data(), 1);
                    manager->engine->convergeState = cs;
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

        unsigned featLayer = chunk.vertex ? chunk.layer : chunk.layer - 1; // YIFAN: fix this
        TensorMap& tensorMap = manager->savedNNTensors[featLayer];
        unsigned more = 1;
        while (more) {
            zmq::message_t tensorHeader(TENSOR_HDR_SIZE);
            workersocket.recv(&tensorHeader);

            std::string name = parseName((char*)tensorHeader.data());
            auto found = tensorMap.find(name);
            if (found == tensorMap.end()) {
                printLog(manager->nodeId, "Requested tensor '%s' not found for layer %u",
                         name.c_str(), featLayer);
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

void LambdaWorker::recvETensors(zmq::message_t& client_id, Chunk& chunk) {
    manager->timeoutMtx.lock();
    bool exist = manager->timeoutTable.find(chunk) != manager->timeoutTable.end();
    manager->timeoutMtx.unlock();
    // printLog(manager->nodeId, "CHecking exist");
    if (exist) {
        int ret = 0;
        unsigned more = 1;
        size_t usize = sizeof(unsigned);
        while (more && ret == 0) {
            // printLog(manager->nodeId, "recv");
            ret = recvETensor(chunk);
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
        // printLog(manager->nodeId, "epoch %u, chunk %u/%u, acc %.3f, loss %.3f", chunk.epoch, accLoss.chunkCnt,
        //     manager->engine->numLambdasForward, accLoss.acc / accLoss.vtcsCnt, accLoss.loss / accLoss.vtcsCnt);
        if (accLoss.chunkCnt == manager->engine->numLambdasForward) {
            printLog(manager->nodeId, "epoch %u, acc %.3f, loss %.3f", chunk.epoch,
                accLoss.acc / accLoss.vtcsCnt, accLoss.loss / accLoss.vtcsCnt);
        }
        manager->accMtx.unlock();
    } else {
        zmq::message_t evalMsg(2 * sizeof(float));
        workersocket.recv(&evalMsg);

        std::string errMsg = "[ ERROR ] when receiving from " + chunk.str() + ": ";
        errMsg += "Received duplicate accloss. Discarding...";
        printLog(manager->nodeId, errMsg.c_str());
    }
}

void LambdaWorker::markFinish(zmq::message_t& client_id, Chunk &chunk) {
    manager->timeoutMtx.lock();
    bool exist = manager->timeoutTable.find(chunk) != manager->timeoutTable.end();
    manager->timeoutMtx.unlock();
    if (exist) {
        zmq::message_t ack(3 * sizeof(unsigned));
        if (manager->NNRecv(chunk)) {
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

// ASSUMPTION: Only one edge tensor requested at a time
void LambdaWorker::sendEdgeTensor(zmq::message_t& client_id, Chunk& chunk) {
    manager->timeoutMtx.lock();
    bool exist = manager->timeoutTable.find(chunk) != manager->timeoutTable.end();
    manager->timeoutMtx.unlock();

    // printLog(manager->nodeId, "Checking exit");
    if (exist) {
        unsigned featLayer = chunk.vertex ? chunk.layer : chunk.layer - 1;
        TensorMap& tMap = manager->savedNNTensors[featLayer];
        zmq::message_t tensorHeader(TENSOR_HDR_SIZE);
        workersocket.recv(&tensorHeader);

        std::string name = parseName((char*)tensorHeader.data());
        auto found = tMap.find(name);
        if (found == tMap.end()) {
            workersocket.send(client_id, ZMQ_SNDMORE);
            printLog(manager->nodeId, "Requested tensor '%s' not found for layer %u",
                     name.c_str(), featLayer);
            zmq::message_t errorHeader(TENSOR_HDR_SIZE);
            populateHeader(errorHeader.data(), ERR_HEADER_FIELD, name.c_str());
            workersocket.send(client_id, ZMQ_SNDMORE);
            workersocket.send(errorHeader);
        } else {
            // printLog(manager->nodeId, "SENDING");
            workersocket.send(client_id, ZMQ_SNDMORE);
            sendEdgeTensorChunk(found->second, chunk);
        }
    } else {
        printLog(manager->nodeId, "Chunk %u DONE", chunk.localId);
        workersocket.send(client_id, ZMQ_SNDMORE);
        zmq::message_t header(TENSOR_HDR_SIZE);
        populateHeader((char*) header.data(), CHUNK_DNE_ERR, CHUNK_DNE_ERR);
        workersocket.send(header);

        char errMsg[1024];
        sprintf(errMsg, "[ ERROR ] when sending chunk: %s %u",
            chunk.str().c_str(), chunk.vertex);
        printLog(manager->nodeId, errMsg);
    }
}

void LambdaWorker::sendEdgeTensorChunk(Matrix& eTensor, Chunk& chunk) {
    CSCMatrix<EdgeType>& csc = (manager->engine->graph).forwardAdj;
    unsigned nChunkEdges = csc.columnPtrs[chunk.upBound] - csc.columnPtrs[chunk.lowBound];
    unsigned long long baseIndex = csc.columnPtrs[chunk.lowBound];

    zmq::message_t responseHeader(TENSOR_HDR_SIZE);
    zmq::message_t edgeDataMsg(nChunkEdges * sizeof(FeatType));
    populateHeader(responseHeader.data(), nChunkEdges, 1);
    std::memcpy(edgeDataMsg.data(), eTensor.getData() + baseIndex, nChunkEdges * sizeof(FeatType));

    workersocket.send(responseHeader, ZMQ_SNDMORE);
    workersocket.send(edgeDataMsg);
}

// JOHN: A lot of information needed for this has to be accessed through engine
//  which is ugly. TODO: Extend matrix class to EdgeMatrix so that all infomration
//  can be encapsulated without accessing engine
void LambdaWorker::sendEdgeInfo(zmq::message_t& client_id, Chunk& chunk) {
    // printLog(manager->nodeId, "SEND EDGE INFO");
    CSCMatrix<EdgeType>& csc = (manager->engine->graph).forwardAdj;
    unsigned numLvids = chunk.upBound - chunk.lowBound;
    unsigned numChunkEdges = csc.columnPtrs[chunk.upBound] - csc.columnPtrs[chunk.lowBound];

    zmq::message_t responseHeader(TENSOR_HDR_SIZE);
    populateHeader(responseHeader.data(), numLvids, numChunkEdges);

    zmq::message_t edgeChunkInfoMsg((numLvids + 1) * sizeof(unsigned long long));
    std::memcpy(edgeChunkInfoMsg.data(), csc.columnPtrs + chunk.lowBound, (numLvids + 1) * sizeof(unsigned long long));
    // std::string colPtrsStr = "Actual colPtrs: ";
    // for (unsigned lvid = chunk.lowBound; lvid <= chunk.upBound; ++lvid) {
    //     colPtrsStr += std::to_string(csc.columnPtrs[lvid]) + " ";
    // }
    // unsigned long long* colPtrMsgData = (unsigned long long*) edgeChunkInfoMsg.data();
    // colPtrsStr += "\ncolPtrData colPtrs: ";
    // for (unsigned lvid = 0; lvid <= numLvids; ++lvid) {
    //     colPtrsStr += std::to_string(colPtrMsgData[lvid]) + " ";
    // }
    workersocket.send(client_id, ZMQ_SNDMORE);
    workersocket.send(responseHeader, ZMQ_SNDMORE);
    workersocket.send(edgeChunkInfoMsg);
    // printLog(manager->nodeId, "MESSGAES SENT FOR EDGE INFO");
}

int LambdaWorker::recvTensor(Chunk &chunk) {
    zmq::message_t tensorHeader(TENSOR_HDR_SIZE);
    workersocket.recv(&tensorHeader);
    zmq::message_t tensorData;
    workersocket.recv(&tensorData);

    std::string name = parseName((char*)tensorHeader.data());
    if (!chunk.vertex)
        return 0;

    unsigned featLayer = chunk.vertex ? chunk.layer : chunk.layer - 1;
    if (manager->engine->gnn_type == GNN::GAT && name == "grad") {
        featLayer = chunk.layer - 1;
    }
    TensorMap& tensorMap = manager->savedNNTensors[featLayer];
    auto found = tensorMap.find(name);
    if (found == tensorMap.end()) {
        printLog(manager->nodeId, "Lambda %s returned unknown tensor %u:'%s'. Make sure to allocate it before running lambdas!",
                 chunk.str().c_str(), featLayer, name.c_str());
        return 1;
    }

    // printLog(manager->nodeId, "get Tensor %s (%u, %u) from %s, dst %s",
    //          name.c_str(), chunk.upBound - chunk.lowBound,
    //          tensorData.size() / (chunk.upBound - chunk.lowBound) / 4,
    //          chunk.str().c_str(), found->second.shape().c_str());
    FeatType* dptr = found->second.get(chunk.lowBound);
    std::memcpy(dptr, tensorData.data(), tensorData.size());

    return 0;
}

int LambdaWorker::recvETensor(Chunk& chunk) {
    zmq::message_t tensorHeader(TENSOR_HDR_SIZE);
    workersocket.recv(&tensorHeader);
    zmq::message_t tensorData;
    workersocket.recv(&tensorData);

    std::string name = parseName((char*)tensorHeader.data());
    unsigned featLayer = chunk.vertex ? chunk.layer : chunk.layer - 1;
    TensorMap& tensorMap = manager->savedNNTensors[featLayer];
    auto found = tensorMap.find(name);
    if (found == tensorMap.end()) {
        printLog(manager->nodeId, "Lambda %s returned unknown tensor '%s'. Make sure to allocate it before running lambdas!",
                 chunk.str().c_str(), name.c_str());
        return 1;
    }

    // printLog(manager->nodeId, "Copying edge values");
    CSCMatrix<EdgeType>& csc = (manager->engine->graph).forwardAdj;
    FeatType* eDptr = found->second.get(csc.columnPtrs[chunk.lowBound]);
    std::memcpy(eDptr, tensorData.data(), tensorData.size());

    return 0;
}
