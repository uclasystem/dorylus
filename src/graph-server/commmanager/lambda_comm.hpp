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


static const bool relaunching = false;

#define LAMBDA_VTX_NN "yifan-gcn"
#define ALLOCATION_TAG "LambdaComm"

class LambdaWorker;
class LockChunkMap;

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
    std::vector<bool> trainPartitions;
    std::vector<TensorMap>& savedNNTensors;
    Lock &resLock;
    ChunkQueue& resQueue;

    std::thread *relaunchThd;
    void asyncRelaunchLoop();
    void relaunchLambda(const Chunk &chunk);
    unsigned relaunchCnt;
    std::mutex tableMtx;
    std::map<Chunk, unsigned> timeoutTable;

    // Invoke lambda function
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
