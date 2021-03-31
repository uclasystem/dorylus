#ifndef __LAMBDA_COMM_HPP__
#define __LAMBDA_COMM_HPP__


#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <mutex>
#include <random>
#include <sstream>
#include <climits>
#include <string>
#include <thread>
#include <vector>
#include <zmq.hpp>

#include <boost/algorithm/string/trim.hpp>

#include <aws/core/Aws.h>
#include <aws/core/utils/json/JsonSerializer.h>
#include <aws/core/utils/Outcome.h>
#include <aws/core/utils/logging/DefaultLogSystem.h>
#include <aws/core/utils/logging/AWSLogging.h>
#include <aws/core/utils/HashingUtils.h>
#include <aws/lambda/LambdaClient.h>
#include <aws/lambda/model/InvokeRequest.h>


#include "resource_comm.hpp"
#include "../engine/engine.hpp"
#include "lambdaworker.hpp"
#include "../utils/utils.hpp"
#include "../parallel/lock.hpp"
#include "../../common/matrix.hpp"
#include "../../common/utils.hpp"


static const bool relaunching = true;

#define ALLOCATION_TAG "LambdaComm"

class LambdaWorker;

/**
 *
 * Class of a lambda threads communication handler.
 *
 */
class LambdaComm : public ResourceComm {
public:
    LambdaComm(Engine *engine);
    ~LambdaComm();

    void NNCompute(Chunk &chunk);
    bool NNRecv(Chunk &chunk);

    unsigned getRelaunchCnt() { return relaunchCnt; };

    bool halt;
    std::vector<TensorMap>& savedNNTensors;
    std::vector<ETensorMap>& savedETensors;

    std::thread *relaunchThd;
    void asyncRelaunchLoop();
    void relaunchLambda(const Chunk &chunk);
    int timeoutRatio;
    unsigned relaunchCnt;
    std::mutex timeoutMtx;
    std::map<Chunk, unsigned> timeoutTable;
    std::map<unsigned, unsigned> recordTable; // map layer -> (avg_time)


    struct AccLoss {
        float acc = 0.0;
        float loss = 0.0;
        unsigned vtcsCnt = 0;
        unsigned chunkCnt = 0;
    };
    std::mutex accMtx;
    std::map<unsigned, AccLoss> accLossTable; // epoch -> AccLoss

    // Invoke lambda function
    std::string LAMBDA_NAME;
    void invokeLambda(const Chunk &chunk);
    static void callback(const Aws::Lambda::LambdaClient *client,
                         const Aws::Lambda::Model::InvokeRequest &invReq,
                         const Aws::Lambda::Model::InvokeOutcome &outcome,
                         const std::shared_ptr<const Aws::Client::AsyncCallerContext> &context);
    zmq::context_t ctx;
    zmq::socket_t frontend;
    zmq::socket_t backend;
    Aws::SDKOptions options;
    std::shared_ptr<Aws::Lambda::LambdaClient> m_client;

    static unsigned nodeId;
    unsigned numNodes;
    // Only for synchronous version //
    unsigned numChunk;
    // ---------------------------- //
    string nodeIp;
    unsigned dport;
    std::vector<std::string> wservers;
    unsigned wport;

    unsigned numListeners;
    std::vector<LambdaWorker *> workers;
    std::vector<std::thread *> worker_threads;
    friend LambdaWorker::LambdaWorker(LambdaComm *manager);

    std::ofstream lambdaOut;

    // Helper utilities
    const char* selectWeightServer(unsigned chunkId);

    Engine *engine;
    void loadWServerIps(std::string wsFile);
    void setupAwsClient();
    void closeAwsClient();
    void setupSockets();
    void closeSockets();
    void createWorkers();
    void stopWorkers();
    void startRelaunchThd();
    void stopRelaunchThd();
};

#endif // LAMBDA_COMM_HPP
