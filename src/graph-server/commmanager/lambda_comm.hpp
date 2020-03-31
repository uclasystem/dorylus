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
#include "../parallel/lock.hpp"
#include "../../common/matrix.hpp"
#include "../../common/utils.hpp"


#define SLEEP_PERIOD 5000   // sleep SLEEP_PERIOD us and then check the condition.
#define TIMEOUT_PERIOD 10000 // wait for up to TIMEOUT_PERIOD ms before relaunching
#define MIN_TIMEOUT 500     // at least wait for MIN_TIMEOUT ms before relaunching
#define EXP_BACKOFF_FACTOR 1.5 // base of exponential backoff

#define FORWARD_FUNC "yifan-forward"
#define BACKWARD_FUNC "yifan-backward"


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
    void invokeLambda(Aws::String funcName, const char* dataserver,
      unsigned dport, char* weightserver, unsigned wport, unsigned layer,
      unsigned id, PROP_TYPE prop_dir, bool lastLayer);
    static void callback(const Aws::Lambda::LambdaClient *client,
      const Aws::Lambda::Model::InvokeRequest &invReq,
      const Aws::Lambda::Model::InvokeOutcome &outcome,
      const std::shared_ptr<const Aws::Client::AsyncCallerContext> &context);

    // Reset LambdaComm state
    void reset(unsigned layer);
    void sendInfoMsg(unsigned layer);

    void newContext(unsigned layer, Matrix &inputTensor_, Matrix &outputTensor_,
                    std::vector<Matrix> *savedTensors_, bool pipeline = false);
    void newContext(unsigned layer, Matrix &inputTensor_, Matrix &outputTensor_, Matrix &targetTensor_,
                    std::vector<Matrix> *savedTensors_, bool pipeline = false);

    // For forward-prop.
    void requestForward(unsigned layer, bool lastLayer);

    void applyVertexForward(unsigned layer, unsigned lambdaId, bool lastLayer);
    void applyEdgeForward(unsigned layer, unsigned lambdaId, bool lastLayer);
    void waitResForward(unsigned layer, bool lastLayer);

    // For backward-prop.
    void sendInfoMessage(zmq::socket_t& wsocket, unsigned numLambdas);
    void requestBackward(unsigned layer, bool lastLayer);

    void applyVertexBackward(unsigned layer, unsigned lambdaId, bool lastLayer);
    void applyEdgeBackward(unsigned layer, unsigned lambdaId, bool lastLayer);
    void waitResBackward(unsigned layer, bool lastLayer);

    void relaunchLambda(bool forward, unsigned layer, unsigned lambdaId, bool lastLayer);
    void relaunchLambda(unsigned layer, unsigned lambdaId, PROP_TYPE prop_dir,
      bool lastLayer);

    void requestInvoke(unsigned layer, unsigned lambdaId, PROP_TYPE prop_dir,
      bool lastLayer);
    void waitLambda(unsigned layer, PROP_TYPE prop_dir, bool lastLayer);

    virtual unsigned getRelaunchCnt() { return relaunchCnt; };

    // Send shutdown messages to the weight servers
    void sendShutdownMessage();

    // simple LambdaWorker initialization
    friend LambdaWorker::LambdaWorker(LambdaComm *manager);

    // data
    Matrix inputTensor;
    Matrix outputTensor;
    Matrix targetTensor;
    std::vector<Matrix> *savedTensors;    // Places to store the intermediate results from lambda.
    std::map<std::string, Matrix>* savedVtxTensors;

    std::vector< TensorMap >* savedNNTensors;

    PairQueue *queuePtr;

// private:
    // AWSSDK Members
    Aws::SDKOptions options;

    unsigned numLambdasForward;
    unsigned numLambdasBackward;
    unsigned numListeners;

    bool pipeline;
    bool forward;
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

    unsigned relaunchCnt;

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
