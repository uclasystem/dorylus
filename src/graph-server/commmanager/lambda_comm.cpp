#include "lambda_comm.hpp"
#include <cmath>

unsigned LambdaComm::nodeId;

/**
 *
 * Lambda communication manager constructor & destructor.
 *
 */
LambdaComm::LambdaComm(Engine *_engine) :
        halt(false), async(false), outdateEpoch(-1),
        nodeIp(_engine->nodeManager.getNode(_engine->nodeId).prip),
        numNodes(_engine->numNodes),
        numChunk(_engine->numLambdasForward),
        relaunchCnt(0),
        dport(_engine->dataserverPort), wport(_engine->weightserverPort),
        savedNNTensors(_engine->savedNNTensors), savedETensors(_engine->savedEdgeTensors),
        resLock(_engine->scatQueueLock), resQueue(_engine->scatterQueue),
        aggLock(_engine->aggQueueLock), aggQueue(_engine->aggregateQueue),
        drvLock(_engine->drvQueueLock), driverQueue(_engine->driverQueue),
        ctx(2), frontend(ctx, ZMQ_ROUTER), backend(ctx, ZMQ_DEALER),
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

void LambdaComm::setAsync(bool _async, unsigned currEpoch) {
    async = _async;
    outdateEpoch = (int)currEpoch;
}

void LambdaComm::NNCompute(Chunk &chunk) {
    // printLog(nodeId, "NNComp: %u:%u:%s", chunk.layer, chunk.localId,
    //   chunk.dir == PROP_TYPE::FORWARD ? "F" : "B");
    timeoutMtx.lock();
    if (timeoutTable.find(chunk) != timeoutTable.end()) {
        printLog(nodeId, "ERROR! duplicated chunk %s", chunk.str().c_str());
    } else {
        timeoutTable[chunk] = timestamp_ms();
        invokeLambda(chunk);
    }
    timeoutMtx.unlock();
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
    // printLog(nodeId, "NNRecv: %u:%u:%s", chunk.layer, chunk.localId,
    //   chunk.dir == PROP_TYPE::FORWARD ? "F" : "B");
    timeoutMtx.lock();
    auto entry = timeoutTable.find(chunk);
    if (entry == timeoutTable.end()) {
        timeoutMtx.unlock();
        // This chunk has already finished
        return false;
    }

    unsigned exeTime = timestamp_ms() - entry->second;
    if (async) engine->vecTimeLambdaWait[chunk.dir * engine->numLayers + chunk.layer] += exeTime;
    timeoutTable.erase(chunk);

    if (chunk.vertex) {
        auto recordFound = recordTable.find(getAbsLayer(chunk, 2));
        if (recordFound == recordTable.end()) {
            recordTable[getAbsLayer(chunk, 2)] = exeTime;
        } else {
            recordTable[getAbsLayer(chunk, 2)] = recordFound->second * 0.95 + exeTime * (1.0 - 0.95);
        }
    }
    timeoutMtx.unlock();

    // filter out outdate chunks after switching sync/async
//    if ((int)(chunk.epoch) <= outdateEpoch) {
//        return true;
//    }

    // If bounded-staleness enabled, increment the finished chunks for this epoch
    // If the min epoch has finished, allow chunks to move to the next epoch
    // TODO (JOHN): Consider using __sync_fetch ops here to make code cleaner and
    //  still achieve synchronization
    if (async) {
        if (isLastLayer(chunk)) {
            if (engine->staleness != UINT_MAX) {
                unsigned ind = chunk.epoch % (engine->staleness + 1);
                engine->finishedChunkLock.lock();
                if (++(engine->numFinishedEpoch[ind]) == numChunk) {
                    engine->finishedChunkLock.unlock();
                    // printLog(nodeId, "FINISHED epoch %u. Total finished %u", chunk.epoch, engine->nodesFinishedEpoch[ind]+1);
                    engine->numFinishedEpoch[ind] = 0;

                    engine->sendEpochUpdate(chunk.epoch);
                    engine->finishedNodeLock.lock();
                    if (++(engine->nodesFinishedEpoch[ind]) == numNodes + 1) {
                        ++(engine->minEpoch);
                        engine->nodesFinishedEpoch[ind] = 0;
                    }
                    engine->finishedNodeLock.unlock();
                } else {
                    engine->finishedChunkLock.unlock();
                }
            }

            chunk.dir = PROP_TYPE::FORWARD;
            chunk.epoch += 1;
            chunk.layer = 0;
            chunk.vertex = true;

            drvLock.lock();
            driverQueue.push(chunk);
            drvLock.unlock();
        } else {
            if (chunk.vertex && chunk.dir == PROP_TYPE::FORWARD) {
                chunk.vertex = !chunk.vertex;
                resLock.lock();
                resQueue.push(chunk);
                resLock.unlock();
            } else if (chunk.vertex && chunk.dir == PROP_TYPE::BACKWARD) {
                chunk.vertex = !chunk.vertex;
                chunk.layer -= 1;
                resLock.lock();
                resQueue.push(chunk);
                resLock.unlock();
            } else if (!chunk.vertex && chunk.dir == PROP_TYPE::FORWARD) {
                chunk.vertex = !chunk.vertex;
                aggLock.lock();
                aggQueue.push(chunk);
                aggLock.unlock();
            } else if (!chunk.vertex && chunk.dir == PROP_TYPE::BACKWARD) {
                chunk.vertex = !chunk.vertex;
                aggLock.lock();
                aggQueue.push(chunk);
                aggLock.unlock();
            }
        }
    } else {
        resLock.lock();
        resQueue.push(chunk);
        resLock.unlock();
    }

    return true;
}

void LambdaComm::asyncRelaunchLoop() {
#define MIN_TIMEOUT 500u     // at least wait for MIN_TIMEOUT ms before relaunching
#define TIMEOUT_PERIOD 3000u // wait for up to TIMEOUT_PERIOD ms before relaunching
#define SLEEP_PERIOD (MIN_TIMEOUT * 1000) // sleep SLEEP_PERIOD us and then check the condition.

    if (!relaunching) return;

    while (!halt) {
        unsigned currTS = timestamp_ms();
        for (auto &kv : timeoutTable) {
            auto found = recordTable.find(getAbsLayer(kv.first, 2));
            if (found == recordTable.end()) {
                continue; // No lambda finished.
            }
            if (kv.first.vertex) {
                if (currTS - kv.second > std::max(MIN_TIMEOUT, 3 * found->second)) {
                    // printLog(nodeId, "curr %u, timed %u, chunk %s", currTS - kv.second, 3 * found->second,
                    //     kv.first.str().c_str());
                    relaunchLambda(kv.first);
                }
            } else {
                if (currTS - kv.second > TIMEOUT_PERIOD) {
                    relaunchLambda(kv.first);
                }
            }
        }
        usleep(SLEEP_PERIOD);
    }

#undef MIN_TIMEOUT
#undef TIMEOUT_PERIOD
#undef SLEEP_PERIOD
}

void LambdaComm::relaunchLambda(const Chunk &chunk) {
    printLog(nodeId, "Relaunch lambda %s for layer %u...", chunk.str().c_str(), chunk.layer);
    timeoutMtx.lock();
    auto entry = timeoutTable.find(chunk);
    if (entry != timeoutTable.end()) {
        entry->second = timestamp_ms();
        ++relaunchCnt;
        invokeLambda(chunk);
    }
    timeoutMtx.unlock();
}

// LAMBDA INVOCATION AND RETURN FUNCTIONS
void LambdaComm::invokeLambda(const Chunk &chunk) {
    Aws::Lambda::Model::InvokeRequest invReq;
    invReq.SetFunctionName(LAMBDA_VTX_NN);
    invReq.SetInvocationType(Aws::Lambda::Model::InvocationType::RequestResponse);
    invReq.SetLogType(Aws::Lambda::Model::LogType::Tail);
    std::shared_ptr<Aws::IOStream> payload = Aws::MakeShared<Aws::StringStream>("LambdaInvoke");

    Aws::Utils::Json::JsonValue jsonPayload;
    jsonPayload.WithString("dserver", nodeIp);
    const char *wserver = selectWeightServer(chunk.localId);
    jsonPayload.WithString("wserver", wserver);
    jsonPayload.WithInteger("dport", dport);
    jsonPayload.WithInteger("wport", wport);
    // jsonPayload.WithBool("eval", (chunk.epoch == 0) || ((chunk.epoch + 1) % 5 == 0));
    jsonPayload.WithBool("eval", true);

    jsonPayload.WithInteger("id", chunk.localId);
    jsonPayload.WithInteger("gid", chunk.globalId);
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
        printLog(nodeId, "\033[1;31m[ ERROR ]\033[0m\treturn error: %s",
                    outcome.GetError().GetMessage().c_str());
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
    clientConfig.region = "us-west-1";
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
