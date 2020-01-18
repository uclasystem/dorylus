#ifndef __ENGINE_HPP__
#define __ENGINE_HPP__


#include <set>
#include <vector>
#include <climits>
#include <atomic>
#include <tuple>
#include <cstdio>
#include "graph.hpp"
#include "../commmanager/commmanager.hpp"
#include "../commmanager/resource_comm.hpp"
#include "../nodemanager/nodemanager.hpp"
#include "../parallel/threadpool.hpp"
#include "../parallel/lock.hpp"
#include "../parallel/cond.hpp"
#include "../parallel/barrier.hpp"
#include "../utils/utils.hpp"
#include "../../common/matrix.hpp"

// Max size (bytes) for a message received by the data communicator.
#define MAX_MSG_SIZE (1024 * 1024)
#define NODE_ID_DIGITS 8 // Digits num of node id.
#define NODE_ID_HEADER "%8X" // Header for node id. For communication.
#define DATA_HEADER_SIZE (NODE_ID_DIGITS + sizeof(unsigned) + sizeof(unsigned))


/** Binary features file header struct. */
struct FeaturesHeaderType {
    unsigned numFeatures;
};

/** Binary labels file header struct. */
struct LabelsHeaderType {
    unsigned labelKinds;
};


/**
 *
 * Class of a GNN-LAMBDA engine executing on a node.
 *
 */
class Engine {

public:

    // Public APIs for benchmarks.
    void init(int argc, char *argv[]);
    FeatType *runForward();
    void runBackward(FeatType *backwardInitData);
    void output();
    void destroy();
    bool master();

    FeatType* aggregate(FeatType *vtcsTensor, unsigned vtcsCnt,
                        unsigned featDim);
    FeatType* invokeLambda(FeatType *vtcsTensor, unsigned vtcsCnt,
                           unsigned inFeatDim, unsigned outFeatDim);
    FeatType* fusedGatherApply(FeatType *vtcsTensor, unsigned vtcsCnt,
                               unsigned inFeatDim, unsigned outFeatDim);
    FeatType* scatter(FeatType *vtcsTensor, unsigned vtcsCnt, unsigned featDim);

    FeatType* aggregateBackward(FeatType *gradTensor, unsigned vtcsCnt,
                                unsigned featDim);
    FeatType* invokeLambdaBackward(FeatType *gradTensor, unsigned vtcsCnt,
                                   unsigned inFeatDim, unsigned outFeatDim);
    FeatType* fusedGatherApplyBackward(FeatType *gradTensor, unsigned vtcsCnt,
                                       unsigned inFeatDim, unsigned outFeatDim);
    FeatType* scatterBackward(FeatType *gradTensor, unsigned vtcsCnt,
                              unsigned featDim);

    void makeBarrier();

    unsigned getNumEpochs();
    unsigned getValFreq();
    unsigned getNodeId();

    void addEpochTime(double epochTime);

private:

    NodeManager nodeManager;
    CommManager commManager;

    Graph graph;

    unsigned dThreads;
    ThreadPool *dataPool = NULL;

    unsigned cThreads;
    ThreadPool *computePool = NULL;

    // Config of number of features in each layer.
    std::vector<unsigned> layerConfig;
    unsigned numLayers = 0;

    // intermediate data for backward computation.
    std::vector<Matrix> *savedTensors;

    FeatType *forwardVerticesInitData;
    FeatType *forwardGhostInitData;

    FeatType *forwardGhostVerticesData;
    FeatType *backwardGhostVerticesData;

    // Labels one-hot storage array.
    FeatType *localVerticesLabels = NULL;

    unsigned currId = 0;

    int recvCnt = 0;
    Lock lockRecvCnt;
    Cond condRecvCnt;

    std::string datasetDir;
    std::string outFile;
    std::string featuresFile;
    std::string layerConfigFile;
    std::string labelsFile;
    std::string dshMachinesFile;
    std::string myPrIpFile;
    std::string myPubIpFile;

    std::time_t start_time;

    unsigned dataserverPort;
    unsigned weightserverPort;
    std::string weightserverIPFile;
    std::string coordserverIp;
    unsigned coordserverPort;

    unsigned numLambdasForward = 0;
    unsigned numLambdasBackward = 0;
    unsigned numEpochs = 0;
    unsigned valFreq = 0;

    float accuracy = 0.0;

    //0: Lambda, 1:GPU, 2: CPU
    unsigned mode = 0;

    ResourceComm *resComm = NULL;
    CommInfo commInfo;
    unsigned nodeId;
    unsigned numNodes;

    bool commHalt = false;

    bool undirected = false;

    unsigned iteration = 0;

    // Timing stuff.
    double timeInit = 0.0;
    double timeForwardProcess = 0.0;
    double timeBackwardProcess = 0.0;
    std::vector<double> vecTimeAggregate;
    std::vector<double> vecTimeLambda;
    std::vector<double> vecTimeSendout;
    std::vector<double> epochTimes;

    Barrier barComp;

    void calcAcc(FeatType *predicts, FeatType *labels, unsigned vtcsCnt,
                 unsigned featDim);

    // Worker and communicator thread function.
    void forwardWorker(unsigned tid, void *args);
    void backwardWorker(unsigned tid, void *args);
    void forwardGhostCommunicator(unsigned tid, void *args);
    void backwardGhostCommunicator(unsigned tid, void *args);

    void aggregateCompute(unsigned tid, void *args);
    void aggregateBPCompute(unsigned tid, void *args);

    void gatherApplyCompute(unsigned tid, void *args);
    void gatherApplyBPCompute(unsigned tid, void *args);

    // About the global data arrays.
    inline unsigned getFeatDim(unsigned layer) {
        return layerConfig[layer];
    }

    inline FeatType *localVertexLabelsPtr(unsigned lvid) {
        return localVerticesLabels + lvid * getFeatDim(numLayers);
    }

    void sendForwardGhostUpdates(FeatType *inputTensor, unsigned featDim);
    void sendBackwardGhostGradients(FeatType *gradTensor, unsigned featDim);
    // Ghost update operation, send vertices to other nodes
    void forwardVerticesPushOut(unsigned receiver, unsigned totCnt,
                                unsigned *lvids, FeatType *inputTensor,
                                unsigned featDim);
    void backwardVerticesPushOut(unsigned receiver, unsigned totCnt,
                                 unsigned *lvids, FeatType *gradTensor,
                                 unsigned featDim);

    // Aggregation operation (along with normalization).
    void forwardAggregateFromNeighbors(unsigned lvid, FeatType *outputTensor,
                                       FeatType *inputTensor, unsigned featDim);
    void backwardAggregateFromNeighbors(unsigned lvid, FeatType *nextGradTensor,
                                        FeatType *gradTensor, unsigned featDim);

    // For initialization.
    void parseArgs(int argc, char* argv[]);
    void readLayerConfigFile(std::string& layerConfigFileName);
    void readFeaturesFile(std::string& featuresFileName);
    void readLabelsFile(std::string& labelsFileName);

    void setUpCommInfo();

    // Metric printing.
    void printGraphMetrics();
    void printEngineMetrics();
};

// Fetch vertex feature from vtx feats array
#define getVtxFeat(dataBuf, lvid, featDim) ((dataBuf) + (lvid) * (featDim))

// Every one includes this file can access the static engine object now
extern Engine engine;

#endif //__ENGINE_HPP__
