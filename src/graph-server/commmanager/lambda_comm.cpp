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
            unsigned rows = parse<unsigned>((char *) header.data(), 2);
            unsigned cols = parse<unsigned>((char *) header.data(), 3);

            switch (op) {
                case (OP::PULL):
                    sendMatrixChunk(worker, identity, partId);
                    break;
                case (OP::PUSH):
                    assert(rows <= actMatrix->getRows());
                    assert(cols == actMatrix->getCols());
                    recvMatrixChunks(worker, identity, partId);
                    break;
                default:
                    printLog(nodeId, "ERROR: Unknown Op code received.\n");
            }
        }
    } catch (std::exception& ex) {
        printLog(nodeId, "ERROR: %s\n", ex.what());
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
        populateHeader((char *) header.data(), -1, -1, -1, -1);
        socket.send(client_id, ZMQ_SNDMORE);
        socket.send(header);

    // Partition id is valid, so send the matrix segment.
    } else {

        // Check to make sure that the bounds of this partition do not exceed the bounds of the data array.
        // If they do, set partition end to the end of the array.
        unsigned partRows = std::ceil((float) actMatrix->getRows() / (float) numLambdas);
        if ((partId * partRows + partRows) > actMatrix->getRows())
            partRows = partRows - (partId * partRows + partRows) + actMatrix->getRows();
        unsigned bufSize = partRows * actMatrix->getCols() * sizeof(FeatType);

        populateHeader((char *) header.data(), OP::RESP, 0, partRows, actMatrix->getCols());
        FeatType *partitionStart = actMatrix->getData() + (partId * partRows * actMatrix->getCols());
        zmq::message_t partitionData(bufSize);
        std::memcpy(partitionData.data(), partitionStart, bufSize);

        socket.send(client_id, ZMQ_SNDMORE);
        socket.send(header, ZMQ_SNDMORE);
        socket.send(partitionData);
    }
}

void
LambdaWorkerForward::recvMatrixChunks(zmq::socket_t& socket, zmq::message_t& client_id, unsigned partId) {
    unsigned partRows = std::ceil((float) actMatrix->getRows() / (float) numLambdas);
    FeatType *partitionZStart = zData + partId * partRows * actMatrix->getCols();
    FeatType *partitionActStart = actData + partId * partRows * actMatrix->getCols();

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


///////////////////////////////////
// Below are LambdaComm methods. //
///////////////////////////////////


/**
 *
 * Call newContextForward() before the lambda invokation to refresh the parameters, and call endContextForward() after
 * lambdas finish to revoke unused memory space (if necessary).
 * 
 */
void
LambdaComm::newContextForward(FeatType *dataBuf, FeatType *zData, FeatType *actData, unsigned numLocalVertices,
                              unsigned numFeats, unsigned numFeatsNext) {
    countForward = 0;

    // Create a new matrix object for workers to access.
    Matrix *actMatrix = new Matrix(numLocalVertices, numFeats, dataBuf);

    // Create a new ZeroMQ context and setup the sockets it holds.
    ctx = new zmq::context_t(1);
    frontend = new zmq::socket_t(*ctx, ZMQ_ROUTER);
    backend = new zmq::socket_t(*ctx, ZMQ_DEALER);
    coordsocket = new zmq::socket_t(*ctx, ZMQ_REQ);

    char dhost_port[50];
    sprintf(dhost_port, "tcp://*:%u", dataserverPort);
    frontend->bind(dhost_port);
    backend->bind("inproc://backend");

    char chost_port[50];
    sprintf(chost_port, "tcp://%s:%u", coordserverIp.c_str(), coordserverPort);
    coordsocket->connect(chost_port);

    // Create 'numListeners' workers and detach them.
    for (unsigned i = 0; i < numListeners; ++i) {
        forwardWorkers.push_back(new LambdaWorkerForward(nodeId, ctx, numLambdasForward, countForward,
                                                         actMatrix, zData, actData, numFeatsNext));
        forwardWorker_threads.push_back(new std::thread(std::bind(&LambdaWorkerForward::work, forwardWorkers[i])));
        forwardWorker_threads[i]->detach();
    }

    // Create a proxy pipe that connects frontend to backend. This thread hangs throughout the lifetime of this context.
    std::thread tproxy([&] {
        try {
            zmq::proxy(static_cast<void *>(*frontend), static_cast<void *>(*backend), nullptr);
        } catch (std::exception& ex) {
            printLog(nodeId, "ERROR: %s\n", ex.what());
        }
    });
    tproxy.detach();

    printLog(nodeId, "Lambda FORWARD context created.\n");
}

void
LambdaComm::endContextForward() {
    countForward = 0;

    // Delete the ZeroMQ context.
    delete ctx;
    delete frontend;
    delete backend;
    delete coordsocket;

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
 * Call newContextForward() before the lambda invokation to refresh the parameters, and call endContextForward() after
 * lambdas finish to revoke unused memory space (if necessary).
 * 
 */
void
LambdaComm::newContextBackward() {
    printLog(nodeId, "Lambda BACKWARD context created.\n");
}

void
LambdaComm::endContextBackward() {
    printLog(nodeId, "Lambda BACKWARD context finished.\n");
}


/**
 *
 * Send a request to the coordination server for a given number of lambda threads to do the forward function.
 * 
 */
void
LambdaComm::requestLambdasForward(unsigned layer) {
    
    // Send header info to tell the coordserver to trigger how many lambdas in which forward layer.
    zmq::message_t header(HEADER_SIZE);
    populateHeader((char *) header.data(), OP::REQ, layer, numLambdasForward);
    coordsocket->send(header, ZMQ_SNDMORE);

    // Send my ip.
    zmq::message_t ip_msg(nodeIp.size());
    std::memcpy(ip_msg.data(), nodeIp.c_str(), nodeIp.size());
    coordsocket->send(ip_msg);
    
    // Wait for a confirm ACK message.
    zmq::message_t confirm;
    coordsocket->recv(&confirm);

    // Block until all parts have been handled.
    std::unique_lock<std::mutex> lk(m);
    cv.wait(lk, [&]{ return countForward == numLambdasForward; });
}


/**
 *
 * Send a request to the coordination server for a given number of lambda threads to do the backward function.
 * 
 */
void
LambdaComm::requestLambdasBackward() {
    
}


/**
 * 
 * Send message to the coordination server to shutdown.
 * 
 */
void
LambdaComm::sendShutdownMessage() {

    // Create a new ZeroMQ context and setup the coord socket.
    ctx = new zmq::context_t(1);
    coordsocket = new zmq::socket_t(*ctx, ZMQ_REQ);

    char chost_port[50];
    sprintf(chost_port, "tcp://%s:%u", coordserverIp.c_str(), coordserverPort);
    coordsocket->connect(chost_port);

    // Send kill message.
    zmq::message_t header(HEADER_SIZE);
    populateHeader((char *) header.data(), OP::TERM);
    coordsocket->send(header, ZMQ_SNDMORE);

    // Send dummy message since coordination server expects an IP as well.
    zmq::message_t dummyIP;
    coordsocket->send(dummyIP);

    delete ctx;
    delete coordsocket;
}
