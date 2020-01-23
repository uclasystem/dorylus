#include "lambda_comm.hpp"
#include <thread>


static std::vector<LambdaWorker *> workers;
static std::vector<std::thread *> worker_threads;


extern "C" ResourceComm* createComm(CommInfo& commInfo) {
    return new LambdaComm(commInfo);
}

extern "C" void destroyComm(LambdaComm *lambdaComm) {
    delete lambdaComm;
}

/**
 *
 * Lambda communication manager constructor & destructor.
 *
 */
LambdaComm::LambdaComm(CommInfo &commInfo) :
        ResourceComm(), nodeIp(commInfo.nodeIp), dataserverPort(commInfo.dataserverPort),
        coordserverIp(commInfo.coordserverIp), coordserverPort(commInfo.coordserverPort),
        nodeId(commInfo.nodeId), numNodes(commInfo.numNodes), ctx(1), halt(false), frontend(ctx, ZMQ_ROUTER), backend(ctx, ZMQ_DEALER), coordsocket(ctx, ZMQ_REQ),
        numLambdasForward(commInfo.numLambdasForward), numLambdasBackward(commInfo.numLambdasBackward), numListeners(numLambdasBackward), // TODO: Decide numListeners.
        countForward(0), countBackward(0), timeoutPeriod(0.0), currLayer(0) {

    // Bind the proxy sockets.
    char dhost_port[50];
    sprintf(dhost_port, "tcp://*:%u", dataserverPort);
    frontend.bind(dhost_port);
    backend.bind("inproc://backend");

    char chost_port[50];
    sprintf(chost_port, "tcp://%s:%u", coordserverIp.c_str(), coordserverPort);
    coordsocket.connect(chost_port);

    // Create 'numListeners' workers and detach them.
    for (unsigned i = 0; i < numListeners; ++i) {
        workers.push_back(new LambdaWorker(this));
        worker_threads.push_back(new std::thread(std::bind(&LambdaWorker::work, workers[i])));
    }

    forwardLambdaTable = new bool[numLambdasForward];
    backwardLambdaTable = new bool[numLambdasBackward];
    memset(forwardLambdaTable, 0, sizeof(bool) * numLambdasForward);
    memset(backwardLambdaTable, 0, sizeof(bool) * numLambdasBackward);

    // Create proxy pipes that connect frontend to backend. This thread hangs throughout the lifetime of this context.
    std::thread tproxy([&] {
        try {
            zmq::proxy(static_cast<void *>(frontend), static_cast<void *>(backend), nullptr);
        } catch (std::exception& ex) { /** Context termintated. */ }
    });
    tproxy.detach();
}

LambdaComm::~LambdaComm() {
    // Delete allocated resources.
    halt = true;
    for (unsigned i = 0; i < numListeners; ++i) {
        // worker_threads[i]->join();
        delete workers[i];
        delete worker_threads[i];
    }

    if (forwardLambdaTable) {
        delete[] forwardLambdaTable;
    }
    if (backwardLambdaTable) {
        delete[] backwardLambdaTable;
    }

    frontend.close();
    backend.close();
    coordsocket.close();

    ctx.close();
}

/**
 *
 * Call 'newContext()' before the lambda invokation to refresh the parameters, then call `requestLambdas()` to tell the coordserver to
 * trigger lambda threads.
 *
 */
void
LambdaComm::newContextForward(unsigned layer, FeatType *dataBuf, FeatType *zData, FeatType *actData,
    unsigned numLocalVertices, unsigned numFeats, unsigned numFeatsNext) {
    countForward = 0;

    currLayer = layer;
    timeoutPeriod = 0.0;

    // Create a new matrix object for workers to access.
    Matrix actMatrix(numLocalVertices, numFeats, dataBuf);

    // Refresh workers' members, and connect their worker sockets to the backend.
    for (auto&& worker : workers)
        worker->refreshState(actMatrix, zData, actData, numFeatsNext);
}

// deprecated.
void
LambdaComm::requestForward(unsigned layer, bool lastLayer) {
    for (unsigned i = 0; i < numLambdasForward; i++) {
        invokeLambdaForward(layer, i, lastLayer);
    }

    waitLambdaForward(layer, lastLayer);
}


void
LambdaComm::invokeLambdaForward(unsigned layer, unsigned lambdaId, bool lastLayer) { // another option is to keep states in coordserver
    zmq::message_t header(HEADER_SIZE);
    populateHeader((char *) header.data(), OP::REQ_FORWARD, layer, nodeId * numLambdasForward + lambdaId, lambdaId, lastLayer);
    coordsocket.send(header, ZMQ_SNDMORE);

    // forwardLambdaTable[lambdaId] = true;
    __sync_bool_compare_and_swap(forwardLambdaTable + lambdaId, false, true);
    if (lambdaId == 0) {
        forwardTimer = getTimer();
    }

    zmq::message_t ip_msg(nodeIp.size());
    std::memcpy(ip_msg.data(), nodeIp.c_str(), nodeIp.size());
    coordsocket.send(ip_msg);

    // Wait for a confirm ACK message.
    zmq::message_t confirm;
    coordsocket.recv(&confirm);
}


void
LambdaComm::waitLambdaForward(unsigned layer, bool lastLayer) {
    // Block until all parts have been handled.
    while (countForward < numLambdasForward) {
        if (countForward >= 0.8 * numLambdasForward && timeoutPeriod < 1e-8) {
            timeoutPeriod = std::fmax(MIN_TIMEOUT, 2 * (getTimer() - forwardTimer));
        }
        if (getTimer() - forwardTimer > (timeoutPeriod < 1e-8 ? TIMEOUT_PERIOD : timeoutPeriod)) {
            for (unsigned i = 0; i < numLambdasForward; i++) {
                if (forwardLambdaTable[i]) {
                    relaunchLambda(true, layer, i, lastLayer);
                }
            }
            forwardTimer = getTimer();
            timeoutPeriod *= EXP_BACKOFF_FACTOR;
        }
        usleep(SLEEP_PERIOD);
    }
}


/**
 *
 * Call 'newContext()' before the lambda invokation to refresh the parameters, then call `requestLambdas()` to tell the coordserver to
 * trigger lambda threads.
 *
 */
void
LambdaComm::newContextBackward(unsigned layer, FeatType *oldGradBuf, FeatType *newGradBuf, std::vector<Matrix> *savedTensors, FeatType *targetBuf, unsigned numLocalVertices, unsigned inFeatDim, unsigned outFeatDim, unsigned targetDim) {
    countBackward = 0;

    currLayer = layer;
    timeoutPeriod = 0.0;

    // Create new matrices object for workers to access.
    Matrix oldGradMatrix(numLocalVertices, outFeatDim, oldGradBuf);
    Matrix newGradMatrix(numLocalVertices, inFeatDim, newGradBuf);
    Matrix targetMatrix(numLocalVertices, targetDim, targetBuf);

    // Refresh workers' members, and connect their worker sockets to the backend.
    for (auto&& worker : workers)
        worker->refreshState(oldGradMatrix, newGradMatrix, targetMatrix, savedTensors);

    // sending backward lambda number to coord server and finally set up weight server.
    if (nodeId == 0) { // I am master and master will send the total number of lambdas of all graph servers.
        zmq::message_t header(HEADER_SIZE);
        populateHeader((char *)header.data(), OP::INFO, numNodes * numLambdasBackward);
        coordsocket.send(header, ZMQ_SNDMORE);
        zmq::message_t dummyIp;
        coordsocket.send(dummyIp);
        zmq::message_t confirm;
        coordsocket.recv(&confirm);
    }

    // printLog(nodeId, "Lambda BACKWARD context created.");
}


void
LambdaComm::requestBackward(unsigned layer, bool lastLayer) {
    for (unsigned i = 0; i < numLambdasBackward; i++) {
        invokeLambdaBackward(layer, i, lastLayer);
    }

    waitLambdaBackward(layer, lastLayer);
}

void
LambdaComm::invokeLambdaBackward(unsigned layer, unsigned lambdaId, bool lastLayer) {
    zmq::message_t header(HEADER_SIZE);
    populateHeader((char *) header.data(), OP::REQ_BACKWARD, layer, nodeId * numLambdasBackward + lambdaId, lambdaId, lastLayer);
    coordsocket.send(header, ZMQ_SNDMORE);

    // backwardLambdaTable[lambdaId] = true;
    __sync_bool_compare_and_swap(backwardLambdaTable + lambdaId, false, true);
    if (lambdaId == 0) {
        backwardTimer = getTimer();
    }

    zmq::message_t ip_msg(nodeIp.size());
    std::memcpy(ip_msg.data(), nodeIp.c_str(), nodeIp.size());
    coordsocket.send(ip_msg);

    // Wait for a confirm ACK message.
    zmq::message_t confirm;
    coordsocket.recv(&confirm);
}

void
LambdaComm::waitLambdaBackward(unsigned layer, bool lastLayer) {
    // Block until all parts have been handled.
    while (countBackward < numLambdasBackward) {
        if (countBackward >= 0.8 * numLambdasBackward && timeoutPeriod < 1e-8) {
            timeoutPeriod = std::fmax(MIN_TIMEOUT, 2 * (getTimer() - backwardTimer));
        }
        if (getTimer() - backwardTimer > (timeoutPeriod < 1e-8 ? TIMEOUT_PERIOD : timeoutPeriod)) {
            for (unsigned i = 0; i < numLambdasBackward; i++) {
                if (backwardLambdaTable[i]) {
                    relaunchLambda(false, layer, i, lastLayer);
                }
            }
            backwardTimer = getTimer();
            timeoutPeriod *= EXP_BACKOFF_FACTOR;
        }
        usleep(SLEEP_PERIOD);
    }
}


void
LambdaComm::relaunchLambda(bool forward, unsigned layer, unsigned lambdaId, bool lastLayer) {
    printLog(nodeId, "Relaunch %s lambda %u...", (forward ? "FORWARD" : "BACKWARD"), lambdaId);
    zmq::message_t header(HEADER_SIZE);
    populateHeader((char *) header.data(), forward ? OP::REQ_FORWARD : OP::REQ_BACKWARD, layer, nodeId * numLambdasBackward, lambdaId, lastLayer);
    coordsocket.send(header, ZMQ_SNDMORE);

    zmq::message_t ip_msg(nodeIp.size());
    std::memcpy(ip_msg.data(), nodeIp.c_str(), nodeIp.size());
    coordsocket.send(ip_msg);

    // Wait for a confirm ACK message.
    zmq::message_t confirm;
    coordsocket.recv(&confirm);
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

    zmq::message_t confirm;
    coordsocket.setsockopt(ZMQ_RCVTIMEO, 500);
    coordsocket.recv(&confirm);
}
