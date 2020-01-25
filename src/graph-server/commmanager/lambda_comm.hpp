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
#include "lambdaworker.hpp"
#include "../utils/utils.hpp"
#include "../../common/matrix.hpp"
#include "../../common/utils.hpp"


#define SLEEP_PERIOD 5000   // sleep 5000us and then check the condition.
#define TIMEOUT_PERIOD 10000 // wait for up to TIMEOUT_PERIOD ms before relaunching
#define MIN_TIMEOUT 500     // at least wait for MIN_TIMEOUT ms before relaunching
#define EXP_BACKOFF_FACTOR 1.5 // base of exponential backoff

class LambdaWorker;

/**
 *
 * Class of a lambda threads communication handler.
 *
 */
class LambdaComm : public ResourceComm {

public:

    LambdaComm(CommInfo &commInfo);
    ~LambdaComm();
    void connectToWeightServers();

    // Invoke lambda function
    void invokeLambda(Aws::String funcName, const char* dataserver,
      unsigned dport, char* weightserver, unsigned wport, unsigned layer,
      unsigned id, bool lastLayer);
    static void callback(const Aws::Lambda::LambdaClient *client,
      const Aws::Lambda::Model::InvokeRequest &invReq,
      const Aws::Lambda::Model::InvokeOutcome &outcome,
      const std::shared_ptr<const Aws::Client::AsyncCallerContext> &context);

    // For forward-prop.
    void newContextForward(unsigned layer, FeatType *dataBuf, FeatType *zData,
      FeatType *actData, unsigned numLocalVertices, unsigned numFeats,
      unsigned numFeatsNext);
    void requestForward(unsigned layer, bool lastLayer);
    void invokeLambdaForward(unsigned layer, unsigned lambdaId, bool lastLayer);
    void waitLambdaForward(unsigned layer, bool lastLayer);

    // For backward-prop.
    void newContextBackward(unsigned layer, FeatType *oldGradBuf,
      FeatType *newGradBuf, std::vector<Matrix> *savedTensors,
      FeatType *targetBuf, unsigned numLocalVertices, unsigned inFeatDim,
      unsigned outFeatDim, unsigned targetDim);
    void sendInfoMessage(zmq::socket_t& wsocket, unsigned numLambdas);
    void requestBackward(unsigned layer, bool lastLayer);
    void invokeLambdaBackward(unsigned layer, unsigned lambdaId, bool lastLayer);
    void waitLambdaBackward(unsigned layer, bool lastLayer);

    void relaunchLambda(bool forward, unsigned layer, unsigned lambdaId, bool lastLayer);

    // Send shutdown messages to the weight servers
    void sendShutdownMessage();

    // simple LambdaWorker initialization
    friend LambdaWorker::LambdaWorker(LambdaComm *manager);

// private:
    // AWSSDK Members
    Aws::SDKOptions options;

    unsigned numLambdasForward;
    unsigned numLambdasBackward;
    unsigned numListeners;

    unsigned currLayer;

    std::string wServersFile;
    std::vector<char*> weightservers;
    unsigned weightserverPort;

    bool halt;
    std::vector<bool> trainPartitions;

    double timeoutPeriod;

    // for relaunch timed-out lambdas
    unsigned countForward;
    bool *forwardLambdaTable;
    double forwardTimer;
    unsigned countBackward;
    bool *backwardLambdaTable;
    double backwardTimer;

    zmq::context_t ctx;
    zmq::socket_t frontend;
    zmq::socket_t backend;
    std::vector<zmq::socket_t> weightsockets;

    unsigned nodeId;
    unsigned numNodes;
    std::string nodeIp;
    unsigned dataserverPort;
};


#endif // LAMBDA_COMM_HPP
