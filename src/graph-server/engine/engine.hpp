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
#include "../commmanager/lambda_comm.hpp"
#include "../commmanager/GPU_comm.hpp"
#include "../nodemanager/nodemanager.hpp"
#include "../parallel/threadpool.hpp"
#include "../parallel/lock.hpp"
#include "../parallel/cond.hpp"
#include "../parallel/barrier.hpp"
#include "../utils/utils.hpp"


#define MAX_MSG_SIZE 8192   // Max size (bytes) for a message received by the data communicator.


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
    static void init(int argc, char *argv[]);
    static void runForward(bool eval = false);
    static void runBackward();
    static void output();
    static void destroy();
    static bool master();
    static bool isGPUEnabled();

    static void makeBarrier();

    static unsigned getNumEpochs();
    static unsigned getValFreq();
    static unsigned getNodeId();

    // For now, split is at a partition granularity
    // For this node, will simply split the data into training data and validaiton data
    // at the partition level (trainPortion = 1/3 means 1/3 of data will be training data
    // and 2/3 will be validation
    // TODO:
    //  optimize this as ML might require things such as random sampling etc for training
    //  QUESTION: Is it beneficial to go beyond parition level for individual vertices
    //      as this will incur serialization overhead
    static void setTrainValidationSplit(float trainPortion);

private:

    static Graph graph;

    static unsigned dThreads;
    static ThreadPool *dataPool;

    static unsigned cThreads;
    static ThreadPool *computePool;

    static std::vector<unsigned> layerConfig;   // Config of number of features in each layer.
    static unsigned numLayers;

    static FeatType **localVerticesZData;       // Global contiguous array for all vertices' data (row-wise order).
    static FeatType **localVerticesActivationData;
    static FeatType **ghostVerticesActivationData;

    static FeatType *localVerticesDataBuf;      // A smaller buffer storing current iter's data after aggregation.

    static FeatType *localVerticesLabels;       // Labels one-hot storage array.

    static unsigned currId;
    static Lock lockCurrId;

    static Lock lockRecvWaiters;
    static Cond condRecvWaitersEmpty;

    static Lock lockHalt;

    static std::string graphFile;
    static std::string outFile;
    static std::string featuresFile;
    static std::string layerConfigFile;
    static std::string labelsFile;
    static std::string dshMachinesFile;
    static std::string myPrIpFile;
    static std::string myPubIpFile;

    static unsigned dataserverPort;
    static std::string coordserverIp;
    static unsigned coordserverPort;

    static unsigned numLambdasForward;
    static unsigned numLambdasBackward;
    static unsigned numEpochs;
    static unsigned valFreq;

    // table representing whether the partition id is a training set or not
    static std::vector<bool> trainPartition;

    static LambdaComm *lambdaComm;
    static unsigned gpuEnabled;
    static GPUComm *gpuComm;
    
    static unsigned nodeId;
    static unsigned numNodes;

    static bool evaluate;

    static bool halt;

    static bool undirected;

    static unsigned iteration;

    // Timing stuff.
    static double timeInit;
    static double timeForwardProcess; 
    static std::vector<double> vecTimeAggregate;
    static std::vector<double> vecTimeLambda;
    static std::vector<double> vecTimeSendout;
    static double timeBackwardProcess; 

    static std::map<unsigned, unsigned> recvWaiters;

    static Barrier barComp;

    // Worker and communicator thread function.
    static void forwardWorker(unsigned tid, void *args);
    static void backwardWorker(unsigned tid, void *args);
    static void ghostCommunicator(unsigned tid, void *args);

    // About the global data arrays.
    static unsigned getNumFeats(unsigned layer);

    static FeatType *localVertexZDataPtr(unsigned lvid, unsigned layer);
    static FeatType *localVertexActivationDataPtr(unsigned lvid, unsigned layer);
    static FeatType *ghostVertexActivationDataPtr(unsigned lvid, unsigned layer);

    static FeatType *localVertexDataBufPtr(unsigned lvid, unsigned layer);
    static FeatType *localVertexLabelsPtr(unsigned lvid);

    // Aggregation operation (along with normalization).
    static void aggregateFromNeighbors(unsigned lvid);

    // For initialization.
    static void parseArgs(int argc, char* argv[]);
    static void readLayerConfigFile(std::string& layerConfigFileName);
    static void readFeaturesFile(std::string& featuresFileName);
    static void readLabelsFile(std::string& labelsFileName);
    static void readPartsFile(std::string& partsFileName, Graph& lGraph);
    static void processEdge(unsigned& from, unsigned& to, Graph& lGraph, std::set<unsigned>* inTopics, std::set<unsigned>* oTopics); 
    static void findGhostDegrees(std::string& fileName);
    static void setEdgeNormalizations();
    static void readGraphBS(std::string& fileName, std::set<unsigned>& inTopics, std::vector<unsigned>& outTopics);

    // Metric printing.
    static void printGraphMetrics();
    static void printEngineMetrics();
};


#endif //__ENGINE_HPP__
