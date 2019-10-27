#include "lambda_comm.hpp"
#include <thread>

std::mutex eval_mutex;
unsigned evalLambdas = 0;

static std::vector<LambdaWorker *> workers;
static std::vector<std::thread *> worker_threads;


extern "C" ResourceComm* createComm(CommInfo& commInfo) {
    return new LambdaComm(commInfo);
}

extern "C" void destoryComm(LambdaComm *lambdaComm) {
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
        nodeId(commInfo.nodeId), ctx(1), halt(false), frontend(ctx, ZMQ_ROUTER), backend(ctx, ZMQ_DEALER), coordsocket(ctx, ZMQ_REQ),
        numLambdasForward(commInfo.numLambdasForward), numLambdasBackward(commInfo.numLambdasBackward), numListeners(numLambdasBackward), // TODO: Decide numListeners.
        countForward(0), countBackward(0), numCorrectPredictions(0), totalLoss(0.0), numValidationVertices(0), evalPartitions(0) {

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
        worker_threads[i]->join();
        delete worker_threads[i];
        delete workers[i];
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
 * Set the training validation split based on the partitions
 * float trainPortion must be between (0,1)
 *
 */
void
LambdaComm::setTrainValidationSplit(float trainPortion, unsigned numLocalVertices) {
    // forward propagation partitioning determined by the number of forward lambdas
    // so assign partitions based on the num of forward lambdas
    unsigned numTrainParts = std::ceil((float)numLambdasForward * trainPortion);

    // NOTE: could be optimized as a bit vector but probaly not that big a deal
    // Set the first 'numTrainParts' partitions to true
    for (unsigned i = 0; i < numLambdasForward; ++i) {
        if (i < numTrainParts)
            trainPartitions.push_back(true);
        else
            trainPartitions.push_back(false);
    }

    // Randomize which partitions are the training ones so it is not always
    // the first 'numTrainParts'
    // COMMENTED OUT FOR DEBUGGING
    // std::random_shuffle(trainPartition.begin(), trainPartition.end());

    // Calculate the total number of validaiton vertices
    // This member is passed by reference to the lambda workers so on update
    // they will have the correct number

    unsigned partVertices = std::ceil((float) numLocalVertices / (float) numLambdasForward);
    for (unsigned i = 0; i < trainPartitions.size(); ++i) {
        if (!trainPartitions[i]) {
            unsigned thisPartVertices = partVertices;
            if ((i * partVertices + partVertices) > numLocalVertices) {
                thisPartVertices = partVertices - (i * partVertices + partVertices) + numLocalVertices;
            }

            numValidationVertices += thisPartVertices;
            ++evalPartitions;
        }
    }
}


/**
 *
 * Call 'newContext()' before the lambda invokation to refresh the parameters, then call `requestLambdas()` to tell the coordserver to
 * trigger lambda threads.
 *
 */
void
LambdaComm::newContextForward(FeatType *dataBuf, FeatType *zData, FeatType *actData,
    unsigned numLocalVertices, unsigned numFeats, unsigned numFeatsNext, bool eval) {
    countForward = 0;
    evaluate = eval;

    // Create a new matrix object for workers to access.
    Matrix actMatrix(numLocalVertices, numFeats, dataBuf);

    // Refresh workers' members, and connect their worker sockets to the backend.
    for (auto&& worker : workers)
        worker->refreshState(actMatrix, zData, actData, numFeatsNext, eval);

    printLog(nodeId, "Lambda FORWARD context created.");
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
    populateHeader((char *) header.data(), OP::REQ_FORWARD, layer, lambdaId, lastLayer);
    coordsocket.send(header, ZMQ_SNDMORE);

    forwardLambdaTable[lambdaId] = true;
    if (lambdaId == numLambdasForward - 1) {
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
        if (getTimer() - forwardTimer > TIMEOUT_PERIOD) {
            for (unsigned i = 0; i < numLambdasForward; i++) {
                if (forwardLambdaTable[i]) {
                    printLog(nodeId, "Relaunch FORWARD lambda %u...", i);
                    invokeLambdaForward(layer, i, lastLayer);
                }
            }
            forwardTimer = getTimer();
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
LambdaComm::newContextBackward(FeatType *oldGradBuf, FeatType *newGradBuf, std::vector<Matrix> *savedTensors, FeatType *targetBuf,
                                unsigned numLocalVertices, unsigned inFeatDim, unsigned outFeatDim, unsigned targetDim) {
    countBackward = 0;

    // Create new matrices object for workers to access.
    Matrix oldGradMatrix(numLocalVertices, outFeatDim, oldGradBuf);
    Matrix newGradMatrix(numLocalVertices, inFeatDim, newGradBuf);
    Matrix targetMatrix(numLocalVertices, targetDim, targetBuf);

    // Refresh workers' members, and connect their worker sockets to the backend.
    for (auto&& worker : workers)
        worker->refreshState(oldGradMatrix, newGradMatrix, targetMatrix, savedTensors);

    printLog(nodeId, "Lambda BACKWARD context created.");
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
    populateHeader((char *) header.data(), OP::REQ_BACKWARD, layer, lambdaId, lastLayer, numLambdasBackward);
    coordsocket.send(header, ZMQ_SNDMORE);

    backwardLambdaTable[lambdaId] = true;
    if (lambdaId == numLambdasBackward - 1) {
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
        if (getTimer() - backwardTimer > TIMEOUT_PERIOD) {
            for (unsigned i = 0; i < numLambdasBackward; i++) {
                if (backwardLambdaTable[i]) {
                    printLog(nodeId, "Relaunch BACKWARD lambda %u...", i);
                    invokeLambdaBackward(layer, i, lastLayer);
                }
            }
            backwardTimer = getTimer();
        }
        usleep(SLEEP_PERIOD);
    }
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
