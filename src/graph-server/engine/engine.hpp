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


#define MAX_MSG_SIZE (256 * 1024)   // Max size (bytes) for a message received by the data communicator.
#define NODE_ID_DIGITS 8 // Digits num of node id.
#define NODE_ID_HEADER "%8X" // Header for node id. For communication.
#define DATA_HEADER_SIZE (NODE_ID_DIGITS + sizeof(unsigned) + sizeof(unsigned))


/** For files cli options. */
#define PARTS_EXT ".parts"
#define EDGES_EXT ".edges"


/** Binary snap file header struct. */
struct BSHeaderType {
    int sizeOfVertexType;
    unsigned numVertices;
    unsigned long long numEdges;
};

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
    FeatType *runForward(bool eval = false);
    void runBackward(FeatType *backwardInitData);
    void output();
    void destroy();
    bool master();
    bool isGPUEnabled();

    FeatType* aggregate(FeatType *vtcsTensor, unsigned vtcsCnt, unsigned featDim);
    FeatType* invokeLambda(FeatType *vtcsTensor, unsigned vtcsCnt, unsigned inFeatDim, unsigned outFeatDim);
    FeatType* scatter(FeatType *vtcsTensor, unsigned vtcsCnt, unsigned featDim);

    FeatType* aggregateBackward(FeatType *gradTensor, unsigned vtcsCnt, unsigned featDim);
    FeatType* invokeLambdaBackward(FeatType *gradTensor, unsigned vtcsCnt, unsigned inFeatDim, unsigned outFeatDim);
    FeatType* scatterBackward(FeatType *gradTensor, unsigned vtcsCnt, unsigned featDim);

    void makeBarrier();

    unsigned getNumEpochs();
    unsigned getValFreq();
    unsigned getNodeId();

    // For now, split is at a partition granularity
    // For this node, will simply split the data into training data and validaiton data
    // at the partition level (trainPortion = 1/3 means 1/3 of data will be training data
    // and 2/3 will be validation
    // TODO:
    //  optimize this as ML might require things such as random sampling etc for training
    //  QUESTION: Is it beneficial to go beyond parition level for individual vertices
    //      as this will incur serialization overhead
    void setTrainValidationSplit(float trainPortion);

private:

    NodeManager nodeManager;
    CommManager commManager;

    Graph graph;

    unsigned dThreads;
    ThreadPool *dataPool = NULL;

    unsigned cThreads;
    ThreadPool *computePool = NULL;

    std::vector<unsigned> layerConfig;      // Config of number of features in each layer.
    unsigned numLayers = 0;

    std::vector<Matrix> *savedTensors; // intermediate data for backward computation.
    FeatType **localVerticesZData;       // Global contiguous array for all vertices' data (row-wise order).
    FeatType **localVerticesActivationData;

    FeatType *aggGradData;
    FeatType *newGradData;

    FeatType *forwardGhostInitData;
    FeatType *forwardGhostVerticesData;
    FeatType *backwardGhostVerticesData;
    unsigned *forwardGhostVCnts;
    unsigned *backwardGhostVCnts;
    unsigned **forwardBatchMsgBuf;
    unsigned **backwardBatchMsgBuf;

    FeatType *localVerticesDataBuf = NULL;  // A smaller buffer storing current iter's data after aggregation.

    FeatType *localVerticesLabels = NULL;   // Labels one-hot storage array.

    unsigned currId = 0;
    Lock lockCurrId;

    int recvCnt = 0;
    Lock lockRecvCnt;
    Cond condRecvCnt;

    Lock lockHalt;

    std::string graphFile;
    std::string outFile;
    std::string featuresFile;
    std::string layerConfigFile;
    std::string labelsFile;
    std::string dshMachinesFile;
    std::string myPrIpFile;
    std::string myPubIpFile;

    unsigned dataserverPort;
    unsigned weightserverPort;
    std::string weightserverIPFile;
    std::string coordserverIp;
    unsigned coordserverPort;

    unsigned numLambdasForward = 0;
    unsigned numLambdasBackward = 0;
    unsigned numEpochs = 0;
    unsigned valFreq = 0;

    // table representing whether the partition id is a training set or not
    std::vector<bool> trainPartition;

    unsigned gpuEnabled = 0;
    ResourceComm *resComm = NULL;
    CommInfo commInfo;
    unsigned nodeId;
    unsigned numNodes;

    bool evaluate = false;

    bool halt = false;
    bool commHalt = false;

    bool undirected = false;

    unsigned iteration = 0;

    // Timing stuff.
    double timeInit = 0.0;
    double timeForwardProcess = 0.0;
    std::vector<double> vecTimeAggregate;
    std::vector<double> vecTimeLambda;
    std::vector<double> vecTimeSendout;
    double timeBackwardProcess = 0.0;

    Barrier barComp;

    // Worker and communicator thread function.
    void forwardWorker(unsigned tid, void *args);
    void backwardWorker(unsigned tid, void *args);
    void forwardGhostCommunicator(unsigned tid, void *args);
    void backwardGhostCommunicator(unsigned tid, void *args);

    void aggregateCompute(unsigned tid, void *args);
    void aggregateBPCompute(unsigned tid, void *args);

    // About the global data arrays.
    inline unsigned getFeatDim(unsigned layer) { return layerConfig[layer]; }

    inline FeatType *localVertexLabelsPtr(unsigned lvid) {
        return localVerticesLabels + lvid * getFeatDim(numLayers);
    }

    void sendForwardGhostUpdates(FeatType *inputTensor, unsigned featDim);
    void sendBackwardGhostGradients(FeatType *gradTensor, unsigned featDim);
    // Ghost update operation, send vertices to other nodes
    void forwardVerticesPushOut(unsigned receiver, unsigned totCnt, unsigned *lvids, FeatType *inputTensor, unsigned featDim);
    void backwardVerticesPushOut(unsigned receiver, unsigned sender, unsigned totCnt, unsigned *lvids, FeatType *gradTensor, unsigned featDim);

    // Aggregation operation (along with normalization).
    void forwardAggregateFromNeighbors(unsigned lvid, FeatType *outputTensor, FeatType *inputTensor, unsigned featDim);
    void backwardAggregateFromNeighbors(unsigned lvid, FeatType *nextGradTensor, FeatType *gradTensor, unsigned featDim);

    // For initialization.
    void parseArgs(int argc, char* argv[]);
    void readLayerConfigFile(std::string& layerConfigFileName);
    void readFeaturesFile(std::string& featuresFileName);
    void readLabelsFile(std::string& labelsFileName);
    void readPartsFile(std::string& partsFileName, Graph& lGraph);
    void processEdge(unsigned& from, unsigned& to, Graph& lGraph, std::set<unsigned>* inTopics, std::set<unsigned>* oTopics, int **forwardGhostVTables, int **backwardGhostVTables);
    void findGhostDegrees(std::string& fileName);
    void setEdgeNormalizations();
    void readGraphBS(std::string& fileName, std::set<unsigned>& inTopics, std::vector<unsigned>& outTopics);
    void setUpCommInfo();

    // Metric printing.
    void printGraphMetrics();
    void printEngineMetrics();
};

// Fetch vertex feature from vtx feats array
#define getVtxFeat(dataBuf, lvid, featDim) ((dataBuf) + (lvid) * (featDim))

// Every one includes this file can access the static engine object now. For TF integration.
extern Engine engine;

#endif //__ENGINE_HPP__
