#include "lambda_comm.hpp"
#include <thread>


std::mutex count_mutex, eval_mutex;
std::condition_variable cv_forward, cv_backward;
unsigned evalLambdas = 0;


static std::vector<LambdaWorker *> workers;
static std::vector<std::thread *> worker_threads;


/**
 *
 * Lambda communication manager constructor & destructor.
 *
 */
LambdaComm::LambdaComm(std::string nodeIp_, unsigned dataserverPort_, std::string coordserverIp_, unsigned coordserverPort_, unsigned nodeId_,
           unsigned numLambdasForward_, unsigned numLambdasBackward_)
    : nodeIp(nodeIp_), dataserverPort(dataserverPort_), coordserverIp(coordserverIp_), coordserverPort(coordserverPort_), nodeId(nodeId_),
      ctx(1), frontend(ctx, ZMQ_ROUTER), backend(ctx, ZMQ_DEALER), coordsocket(ctx, ZMQ_REQ),
      numLambdasForward(numLambdasForward_), numLambdasBackward(numLambdasBackward_), numListeners(numLambdasBackward_),   // TODO: Decide numListeners.
      countForward(0), countBackward(0),
      numCorrectPredictions(0), totalLoss(0.0), numValidationVertices(0), evalPartitions(0) {

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
        worker_threads[i]->detach();
    }

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
    for (unsigned i = 0; i < numListeners; ++i) {
        delete workers[i];
        delete worker_threads[i];
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
//    std::random_shuffle(trainPartition.begin(), trainPartition.end());

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


void
LambdaComm::invokeLambdaForward(unsigned layer, unsigned lambdaId) { // another option is to keep status in coordserver
    zmq::message_t header(HEADER_SIZE);
    populateHeader((char *) header.data(), OP::REQ_FORWARD, layer, lambdaId);
    coordsocket.send(header, ZMQ_SNDMORE);

    zmq::message_t ip_msg(nodeIp.size());
    std::memcpy(ip_msg.data(), nodeIp.c_str(), nodeIp.size());
    coordsocket.send(ip_msg);

    // Wait for a confirm ACK message.
    zmq::message_t confirm;
    coordsocket.recv(&confirm);
}


void
LambdaComm::waitLambdaForward() {
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
LambdaComm::newContextBackward(FeatType **zBufs, FeatType **actBufs, FeatType *targetBuf,
                               unsigned numLocalVertices, std::vector<unsigned> layerConfig) {
    countBackward = 0;

    // Create new matrix objects for workers to access.
    std::vector<Matrix> zMatrices;
    for (size_t i = 1; i < layerConfig.size(); ++i)
        zMatrices.push_back(Matrix(numLocalVertices, layerConfig[i], zBufs[i]));

    std::vector<Matrix> actMatrices;
    for (size_t i = 0; i < layerConfig.size(); ++i)
        actMatrices.push_back(Matrix(numLocalVertices, layerConfig[i], actBufs[i]));

    Matrix targetMatrix(numLocalVertices, layerConfig[layerConfig.size() - 1], targetBuf);

    // Refresh workers' members.
    for (auto&& worker : workers)
        worker->refreshState(zMatrices, actMatrices, targetMatrix);

    printLog(nodeId, "Lambda BACKWARD context created.");
}

void
LambdaComm::requestLambdasBackward(unsigned numLayers_) {

    // Send header info to tell the coordserver to trigger how many lambdas to trigger.
    zmq::message_t header(HEADER_SIZE);
    populateHeader((char *) header.data(), OP::REQ_BACKWARD, numLayers_, numLambdasBackward);
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
    cv_backward.wait(lk, [&]{ return countBackward == numLambdasBackward; });
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
