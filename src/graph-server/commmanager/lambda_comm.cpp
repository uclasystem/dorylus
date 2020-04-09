#include "lambda_comm.hpp"

unsigned LambdaComm::nodeId;

/**
 *
 * Lambda communication manager constructor & destructor.
 *
 */
LambdaComm::LambdaComm(Engine *_engine) :
        halt(false),
        nodeIp(_engine->nodeManager.getNode(_engine->nodeId).prip),
        numNodes(_engine->numNodes),
        numChunk(_engine->numLambdasForward),
        relaunchCnt(0), async(false),
        dport(_engine->dataserverPort), wport(_engine->weightserverPort),
        // TODO: (YIFAN) WARNING!!! Here we should push results back to scatterQueue.
        // Since we don't have scatter now, I set it to aggregateQueue for testing.
        // Change both resLock and resQueue to scatter ones later!
        savedNNTensors(_engine->savedNNTensors), resLock(_engine->consumerQueueLock),
        resQueue(_engine->scatterQueue),
        aggLock(_engine->aggregateConsumerLock), aggQueue(_engine->aggregateQueue),
        ctx(4), frontend(ctx, ZMQ_ROUTER), backend(ctx, ZMQ_DEALER),
        numListeners(4), engine(_engine) { // TODO: Decide numListeners.
    nodeId = _engine->nodeId;

    loadWServerIps(_engine->weightserverIPFile);
    setupAwsClient();
    setupSockets();
    createWorkers();
    startRelaunchThd();

    lambdaOut = std::ofstream(std::string("lambdats") + std::to_string(numChunk) + std::string(".txt"), std::ofstream::out);
}

LambdaComm::~LambdaComm() {
    halt = true;
    lambdaOut.close();
    // Delete allocated resources.
    stopRelaunchThd();
    closeSockets();
    stopWorkers();
    closeAwsClient();
    ctx.close();
}

void LambdaComm::setAsync(bool _async) {
    async = _async;
}

void LambdaComm::NNCompute(Chunk &chunk) {
    printLog(nodeId, "NNComp: %u:%u:%s", chunk.layer, chunk.chunkId,
      chunk.dir == PROP_TYPE::FORWARD ? "F" : "B");
    tableMtx.lock();
    timeoutTable[chunk] = timestamp_ms();
    tableMtx.unlock();
    invokeLambda(chunk);
}

void LambdaComm::NNSync() {
    while (resQueue.size() != engine->numLambdasForward) {
        usleep(20*1000);
    }

    resLock.lock();
    for (unsigned u = 0; u < engine->numLambdasForward; ++u) {
        resQueue.pop();
    }
    resLock.unlock();
}

bool LambdaComm::NNRecv(Chunk &chunk) {
    printLog(nodeId, "NNRecv: %u:%u:%s", chunk.layer, chunk.chunkId,
      chunk.dir == PROP_TYPE::FORWARD ? "F" : "B");
    tableMtx.lock();
    if (timeoutTable.find(chunk) == timeoutTable.end()) {
        tableMtx.unlock();
        // This chunk has already finished
        return false;
    } else {
        timeoutTable.erase(chunk);
        tableMtx.unlock();

        // If final layer, switch direction
        if (chunk.layer == 1) chunk.dir = PROP_TYPE::BACKWARD;
        if (chunk.dir == PROP_TYPE::FORWARD) chunk.layer += 1;
        if (chunk.dir == PROP_TYPE::BACKWARD) chunk.layer -= 1;
        resLock.lock();
        resQueue.push(chunk);
        resLock.unlock();
    }
    return true;
}

bool LambdaComm::enqueueAggChunk(Chunk& chunk) {
    printLog(nodeId, "NNEnq: %u:%u", chunk.layer, chunk.chunkId,
      chunk.dir == PROP_TYPE::FORWARD ? "F" : "B");
    tableMtx.lock();
    if (timeoutTable.find(chunk) == timeoutTable.end()) {
        tableMtx.unlock();
        return false;
    } else {
        timeoutTable.erase(chunk);
        tableMtx.unlock();

        chunk.layer = 0;
        chunk.dir = PROP_TYPE::FORWARD;
        chunk.epoch += 1;
        aggLock.lock();
        aggQueue.push(chunk);
        aggLock.unlock();
    }

    return true;
}

void LambdaComm::asyncRelaunchLoop() {
#define SLEEP_PERIOD 500000 // sleep SLEEP_PERIOD us and then check the condition.
#define TIMEOUT_PERIOD 3000 // wait for up to TIMEOUT_PERIOD ms before relaunching
#define MIN_TIMEOUT 500     // at least wait for MIN_TIMEOUT ms before relaunching
#define EXP_BACKOFF_FACTOR 1.5 // base of exponential backoff

    if (!relaunching) return;

    while (!halt) {
        unsigned currTS = timestamp_ms();
        for (auto &kv : timeoutTable) {
            if (currTS - kv.second > TIMEOUT_PERIOD) {
                relaunchLambda(kv.first);
            }
        }
        usleep(SLEEP_PERIOD);
    }

#undef SLEEP_PERIOD
#undef TIMEOUT_PERIOD
#undef MIN_TIMEOUT
#undef EXP_BACKOFF_FACTOR
}

void LambdaComm::relaunchLambda(const Chunk &chunk) {
    printLog(nodeId, "Relaunch lambda %u for layer %u...", chunk.chunkId, chunk.layer);
    tableMtx.lock();
    timeoutTable[chunk] = timestamp_ms();
    ++relaunchCnt;
    tableMtx.unlock();
    invokeLambda(chunk);
}

// LAMBDA INVOCATION AND RETURN FUNCTIONS
void LambdaComm::invokeLambda(const Chunk &chunk) {
    Aws::Lambda::Model::InvokeRequest invReq;
    if (chunk.vertex) { // vertex NN
        invReq.SetFunctionName(LAMBDA_VTX_NN);
    } else {
        // TODO: set edge NN func name here
    }
    invReq.SetInvocationType(Aws::Lambda::Model::InvocationType::RequestResponse);
    invReq.SetLogType(Aws::Lambda::Model::LogType::Tail);
    std::shared_ptr<Aws::IOStream> payload = Aws::MakeShared<Aws::StringStream>("LambdaInvoke");

    Aws::Utils::Json::JsonValue jsonPayload;
    jsonPayload.WithString("dserver", nodeIp);
    const char *wserver = selectWeightServer(chunk.chunkId);
    jsonPayload.WithString("wserver", wserver);
    jsonPayload.WithInteger("dport", dport);
    jsonPayload.WithInteger("wport", wport);

    jsonPayload.WithInteger("id", chunk.chunkId);
    jsonPayload.WithInteger("lb", chunk.lowBound);
    jsonPayload.WithInteger("ub", chunk.upBound);
    jsonPayload.WithInteger("layer", chunk.layer);
    jsonPayload.WithInteger("dir", chunk.dir);
    jsonPayload.WithInteger("epoch", chunk.epoch);
    jsonPayload.WithInteger("vtx", chunk.vertex);

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
        Aws::IOStream& payload = result.GetPayload();
        Aws::String funcErr = result.GetFunctionError();
        Aws::String resultStr;
        std::getline(payload, resultStr);
        // JSON Parsing not working from Boost to AWS.
        Aws::Utils::Json::JsonValue response(resultStr);
        auto v = response.View();
        if (funcErr != "") {
            Aws::IOStream& requestBody = *(invReq.GetBody());
            Aws::String requestStr;
            std::getline(requestBody, requestStr);
            if (v.KeyExists("errorMessage")) {
                printLog(nodeId, "\033[1;31m[ FUNC ERROR ]\033[0m %s, %s", funcErr.c_str(), v.GetString("errorMessage").c_str());
            } else {
                printLog(nodeId, "\033[1;31m[ FUNC ERROR ]\033[0m %s, %s", funcErr.c_str(), resultStr.c_str());
            }
        } else {
            if (v.KeyExists("success")) {
                if (v.GetBool("success")) {
                } else {
                    if (v.KeyExists("reason")) {
                        printLog(nodeId, "\033[1;31m[ ERROR ]\033[0m\tReason: %s", v.GetString("reason").c_str());
                    }
                }
            } else {
                printLog(nodeId, "\033[1;31m[ ERROR ]\033[0m\tUnable to parse: %s", resultStr.c_str());
            }
        }
    // Lambda returns error.
    } else {
        printLog(nodeId, "\033[1;31m[ ERROR ]\033[0m\treturn error.");
    }
}
// END LAMBDA INVOCATION AND RETURN FUNCTIONS

// Helper functions
const char* LambdaComm::selectWeightServer(unsigned chunkId) {
    // TODO: (YIFAN) Here I use engine.numLambdaForward to automatically adjust to engine's setting. This is ugly.
    return wservers[(nodeId * engine->numLambdasForward + chunkId) % wservers.size()].c_str();
}

void LambdaComm::setupAwsClient() {
    Aws::InitAPI(options);
    Aws::Client::ClientConfiguration clientConfig;
    clientConfig.requestTimeoutMs = 900000;
    clientConfig.maxConnections = 1000;
    clientConfig.region = "us-east-2";
    m_client = Aws::MakeShared<Aws::Lambda::LambdaClient>(ALLOCATION_TAG,
                                                          clientConfig);
}

void LambdaComm::closeAwsClient() {
    m_client = nullptr;
    Aws::ShutdownAPI(options);
}

void LambdaComm::setupSockets() {
    // Bind the proxy sockets.
    char dhost_port[50];
    sprintf(dhost_port, "tcp://*:%u", dport);
    frontend.setsockopt(ZMQ_BACKLOG, 1000);
    frontend.setsockopt(ZMQ_RCVHWM, 5000);
    frontend.setsockopt(ZMQ_SNDHWM, 5000);
    frontend.setsockopt(ZMQ_ROUTER_MANDATORY, 1);
    frontend.bind(dhost_port);
    backend.setsockopt(ZMQ_BACKLOG, 1000);
    backend.setsockopt(ZMQ_RCVHWM, 5000);
    backend.setsockopt(ZMQ_SNDHWM, 5000);
    backend.bind("inproc://backend");
}

void LambdaComm::loadWServerIps(std::string wsFile) {
    std::ifstream infile(wsFile);
    if (!infile.good())
        fprintf(stderr, "Cannot open weight server file: %s [Reason: %s]\n",
                 wsFile.c_str(), std::strerror(errno));
    assert(infile.good());

    std::string line;
    while (!infile.eof()) {
        std::getline(infile, line);
        boost::algorithm::trim(line);

        if (line.length() == 0)
            continue;
        char *addr = strdup(line.c_str());
        wservers.push_back(addr);
    }
}

void LambdaComm::closeSockets() {
    frontend.setsockopt(ZMQ_LINGER, 0);
    frontend.close();
    backend.setsockopt(ZMQ_LINGER, 0);
    backend.close();
}

void LambdaComm::createWorkers() {
    // Create 'numListeners' workers and detach them.
    for (unsigned i = 0; i < numListeners; ++i) {
        workers.push_back(new LambdaWorker(this));
        worker_threads.push_back(new std::thread(std::bind(&LambdaWorker::work, workers[i], i)));
    }

    // Create proxy pipes that connect frontend to backend. This thread hangs throughout the lifetime of this context.
    std::thread tproxy([&] {
        try {
            zmq::proxy(static_cast<void *>(frontend), static_cast<void *>(backend), nullptr);
        } catch (std::exception& ex) { /** Context termintated. */ }
    });
    tproxy.detach();
}

void LambdaComm::stopWorkers() {
    for (unsigned i = 0; i < numListeners; ++i) {
        worker_threads[i]->join();
        delete worker_threads[i];
        delete workers[i];
    }
}

void LambdaComm::startRelaunchThd() {
    relaunchThd = new std::thread(std::bind(&LambdaComm::asyncRelaunchLoop, this));
}

void LambdaComm::stopRelaunchThd() {
    relaunchThd->join();
    delete relaunchThd;
}
