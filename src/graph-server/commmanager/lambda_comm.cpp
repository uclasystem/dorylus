#include "lambda_comm.hpp"


static std::vector<LambdaWorker *> workers;
static std::vector<std::thread *> worker_threads;

std::mutex producerQueueLock;


extern "C" ResourceComm* createComm(CommInfo& commInfo) {
    return new LambdaComm(commInfo);
}

extern "C" void destroyComm(LambdaComm *lambdaComm) {
    delete lambdaComm;
}

const char* ALLOCATION_TAG = "LambdaComm";
static std::shared_ptr<Aws::Lambda::LambdaClient> m_client;
static unsigned globalNodeId = -1;

/**
 *
 * Lambda communication manager constructor & destructor.
 *
 */
LambdaComm::LambdaComm(CommInfo &commInfo) :
        ResourceComm(), nodeIp(commInfo.nodeIp), dataserverPort(commInfo.dataserverPort),
        wServersFile(commInfo.wServersFile), weightserverPort(commInfo.weightserverPort),
        nodeId(commInfo.nodeId), numNodes(commInfo.numNodes), ctx(1), halt(false), frontend(ctx, ZMQ_ROUTER), backend(ctx, ZMQ_DEALER),
        numLambdasForward(commInfo.numLambdasForward), numLambdasBackward(commInfo.numLambdasBackward), numListeners(numLambdasBackward), // TODO: Decide numListeners.
        countForward(0), countBackward(0), timeoutPeriod(0.0), currLayer(0) {
    // If master, establish connections to weight servers
    connectToWeightServers();

    Aws::InitAPI(options);
    Aws::Client::ClientConfiguration clientConfig;
    clientConfig.requestTimeoutMs = 900000;
    clientConfig.region = "us-east-2";
    m_client = Aws::MakeShared<Aws::Lambda::LambdaClient>(ALLOCATION_TAG,
                                                          clientConfig);
    globalNodeId = nodeId;

    // Bind the proxy sockets.
    char dhost_port[50];
    sprintf(dhost_port, "tcp://*:%u", dataserverPort);
    frontend.bind(dhost_port);
    backend.bind("inproc://backend");

    // Create 'numListeners' workers and detach them.
    for (unsigned i = 0; i < numListeners; ++i) {
        workers.push_back(new LambdaWorker(this, commInfo.queuePtr));
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

    m_client = nullptr;
    Aws::ShutdownAPI(options);

    if (forwardLambdaTable) {
        delete[] forwardLambdaTable;
    }
    if (backwardLambdaTable) {
        delete[] backwardLambdaTable;
    }

    frontend.close();
    backend.close();

    ctx.close();
}

void LambdaComm::connectToWeightServers() {
    std::ifstream infile(wServersFile);
    if (!infile.good())
        printLog(nodeId, "Cannot open weight server file: %s [Reason: %s]\n",
                 wServersFile.c_str(), std::strerror(errno));

    assert(infile.good());

    std::string line;
    while (!infile.eof()) {
        std::getline(infile, line);
        boost::algorithm::trim(line);

        if (line.length() == 0)
            continue;

        char *addr = strdup(line.c_str());

        if (nodeId == 0) {
            // While reading in string, also initialize connection if master
            unsigned ind = weightservers.size();
            weightsockets.push_back(zmq::socket_t(ctx, ZMQ_DEALER));
            char identity[] = "graph-master";
            weightsockets[ind].setsockopt(ZMQ_IDENTITY, identity, strlen(identity) + 1);
            char whost_port[50];
            sprintf(whost_port, "tcp://%s:%u", addr, weightserverPort);
            weightsockets[ind].connect(whost_port);
        }

        weightservers.push_back(addr);
    }
}

// LAMBDA INVOCATION AND RETURN FUNCTIONS
void LambdaComm::invokeLambda(Aws::String funcName, const char* dataserver, unsigned dport,
  char* weightserver, unsigned wport, unsigned layer, unsigned id,
  bool lastLayer) {
    Aws::Lambda::Model::InvokeRequest invReq;
    invReq.SetFunctionName(funcName);
    invReq.SetInvocationType(Aws::Lambda::Model::InvocationType::RequestResponse);
    invReq.SetLogType(Aws::Lambda::Model::LogType::Tail);
    std::shared_ptr<Aws::IOStream> payload = Aws::MakeShared<Aws::StringStream>("LambdaInvoke");

    Aws::Utils::Json::JsonValue jsonPayload;
    jsonPayload.WithString("dataserver", dataserver);
    jsonPayload.WithString("weightserver", weightserver);
    jsonPayload.WithInteger("wport", wport);
    jsonPayload.WithInteger("dport", dport);
    jsonPayload.WithInteger("layer", layer);    // For forward-prop: layer-ID; For backward-prop: numLayers.
    jsonPayload.WithInteger("id", id);
    jsonPayload.WithBool("lastLayer", lastLayer);
    *payload << jsonPayload.View().WriteReadable();
    invReq.SetBody(payload);
    m_client->InvokeAsync(invReq, callback);
}


void LambdaComm::callback(const Aws::Lambda::LambdaClient *client,
  const Aws::Lambda::Model::InvokeRequest &invReq,
  const Aws::Lambda::Model::InvokeOutcome &outcome,
  const std::shared_ptr<const Aws::Client::AsyncCallerContext> &context) {
    if (outcome.IsSuccess()) {
        Aws::Lambda::Model::InvokeResult& result = const_cast<Aws::Lambda::Model::InvokeResult&>(outcome.GetResult());

        // JSON Parsing not working from Boost to AWS.
        Aws::IOStream& payload = result.GetPayload();
        Aws::String resultStr;
        std::getline(payload, resultStr);
        Aws::Utils::Json::JsonValue response(resultStr);

        char logMsg[256];
        auto v = response.View();
        if (v.GetBool("success")) {
            if (v.GetInteger("type") == PROP_TYPE::FORWARD) {
                sprintf(logMsg, "END FORWARD %u %u %u %u %u %u",
                  v.GetInteger("id"), v.GetInteger("start"),
                  v.GetInteger("reqStart"), v.GetInteger("reqStart"),
                  v.GetInteger("sendStart"), v.GetInteger("sendEnd"));
            } else {
                sprintf(logMsg, "END BACKWARD %u %u %u %u %u %u %u %u %u %u",
                  v.GetInteger("id"), v.GetInteger("start"),
                  v.GetInteger("reqT0Start"), v.GetInteger("reqT0End"),
                  v.GetInteger("reqT1Start"), v.GetInteger("reqT1End"),
                  v.GetInteger("reqT2Start"), v.GetInteger("reqT2End"),
                  v.GetInteger("sendStart"), v.GetInteger("sendEnd"));
            }
        } else {
            printLog(globalNodeId, "\033[1;31m[ ERROR ]\033[0m\t%s\n", v.GetString("reason").c_str());
        }
    // Lambda returns error.
    } else {
        printLog(globalNodeId, "\033[1;31m[ ERROR ]\033[0m");
    }
}
// END LAMBDA INVOCATION AND RETURN FUNCTIONS


/**
 *
 * Call 'newContext()' before the lambda invokation to refresh the parameters, then call `requestLambdas()`
 *
 */
void
LambdaComm::newContextForward(unsigned layer, FeatType *dataBuf, FeatType *zData,
  FeatType *actData, unsigned numLocalVertices, unsigned numFeats,
  unsigned numFeatsNext, bool pipeline) {
    countForward = 0;

    currLayer = layer;
    timeoutPeriod = 0.0;

    // Create a new matrix object for workers to access.
    Matrix actMatrix(numLocalVertices, numFeats, dataBuf);

    // Refresh workers' members, and connect their worker sockets to the backend.
    for (auto&& worker : workers)
        worker->refreshState(actMatrix, zData, actData, numFeatsNext, pipeline);
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
LambdaComm::invokeLambdaForward(unsigned layer, unsigned lambdaId, bool lastLayer) {
    // forwardLambdaTable[lambdaId] = true;
    __sync_bool_compare_and_swap(forwardLambdaTable + lambdaId, false, true);
    if (lambdaId == 0) {
        forwardTimer = getTimer();
    }

    char* weightServerIp = weightservers[(nodeId * numLambdasForward + lambdaId) % weightservers.size()];
    invokeLambda("eval-forward-gcn", nodeIp.c_str(), dataserverPort, weightServerIp, weightserverPort, layer, lambdaId, lastLayer);
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
 * Call 'newContext()' before the lambda invokation to refresh the parameters, then call `requestLambdas()`
 *
 */
void
LambdaComm::newContextBackward(unsigned layer, FeatType *oldGradBuf, FeatType *newGradBuf, std::vector<Matrix> *savedTensors, FeatType *targetBuf, unsigned numLocalVertices, unsigned inFeatDim, unsigned outFeatDim, unsigned targetDim, bool pipeline) {
    countBackward = 0;

    currLayer = layer;
    timeoutPeriod = 0.0;

    // Create new matrices object for workers to access.
    Matrix oldGradMatrix(numLocalVertices, outFeatDim, oldGradBuf);
    Matrix newGradMatrix(numLocalVertices, inFeatDim, newGradBuf);
    Matrix targetMatrix(numLocalVertices, targetDim, targetBuf);

    // Refresh workers' members, and connect their worker sockets to the backend.
    for (auto&& worker : workers)
        worker->refreshState(oldGradMatrix, newGradMatrix, targetMatrix, savedTensors, pipeline);

    if (nodeId == 0) { // I am master and master will send the total number of lambdas of all graph servers.
        unsigned numLambdas = numNodes * numLambdasBackward;

        unsigned baseNumThreads = numLambdas / weightservers.size();
        unsigned remainder = numLambdas % weightservers.size();

        for (unsigned u = 0; u < remainder; ++u) {
            sendInfoMessage(weightsockets[u % weightservers.size()], baseNumThreads + 1);
        }

        for (unsigned u = remainder; u < weightservers.size(); ++u) {
            sendInfoMessage(weightsockets[u % weightservers.size()], baseNumThreads);
        }
    }
}

void
LambdaComm::sendInfoMessage(zmq::socket_t& wsocket, unsigned numLambdas) {
    zmq::message_t info_header(HEADER_SIZE);
    populateHeader((char*) info_header.data(), OP::INFO, numLambdas);
    wsocket.send(info_header);

    zmq::message_t ack;
    wsocket.recv(&ack);
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
    // backwardLambdaTable[lambdaId] = true;
    __sync_bool_compare_and_swap(backwardLambdaTable + lambdaId, false, true);
    if (lambdaId == 0) {
        backwardTimer = getTimer();
    }

    char* weightServerIp = weightservers[(nodeId * numLambdasForward + lambdaId) % weightservers.size()];
    invokeLambda("eval-backward-gcn", nodeIp.c_str(), dataserverPort, weightServerIp, weightserverPort, layer, lambdaId, lastLayer);
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

    Aws::String funcName = forward ? "eval-forward-gcn" : "eval-backward-gcn";
    unsigned numLambdas = forward ? numLambdasForward : numLambdasBackward;
    char* weightServerIp = weightservers[(nodeId * numLambdas + lambdaId) % weightservers.size()];
    invokeLambda(funcName, nodeIp.c_str(), dataserverPort, weightServerIp,
                 weightserverPort, layer, lambdaId, lastLayer);
}


/**
 *
 * Send shutdown messages to the weight servers
 *
 */
void
LambdaComm::sendShutdownMessage() {
    // Send kill message.
    if (nodeId == 0) {
        printLog(nodeId, "Terminating weight servers");

        for (zmq::socket_t& wsocket : weightsockets) {
            zmq::message_t header(HEADER_SIZE);
            populateHeader((char*) header.data(), OP::TERM);
            wsocket.send(header);

            zmq::message_t ack;
            wsocket.recv(&ack);
        }
    }
}
