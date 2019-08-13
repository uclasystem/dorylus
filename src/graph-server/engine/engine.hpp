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
    IdType numVertices;
    unsigned long long numEdges;
};

/** Binary features file header struct. */
struct FeaturesHeaderType {
    unsigned numFeatures;
};

/** Binary labels file header struct. */
struct LabelsHeaderType {
    LabelType labelKinds;
};


/**
 *
 * Class of an ASPIRE engine executing on a node.
 * 
 */
class Engine {

public:

    // Public APIs for benchmarks.
    static void init(int argc, char *argv[]);
    static void run();
    static void output();
    static void destroy();
    static bool master();

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

    static IdType currId;
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

    static LambdaComm *lambdaComm;

    static unsigned nodeId;
    static unsigned numNodes;

    static bool halt;

    static bool undirected;

    static unsigned iteration;

    // Timing stuff.
    static double timeInit;
    static double timeProcess; 
    static double timeOutput; 
    static std::vector<double> vecTimeAggregate;
    static std::vector<double> vecTimeLambda;
    static std::vector<double> vecTimeSendout;

    static std::map<IdType, unsigned> recvWaiters;

    static Barrier barComp;

    // Worker and communicator thread function.
    static void worker(unsigned tid, void *args);
    static void dataCommunicator(unsigned tid, void *args);

    // About the global data arrays.
    static unsigned getNumFeats(unsigned layer);

    static FeatType *localVertexZDataPtr(IdType lvid, unsigned layer);
    static FeatType *localVertexActivationDataPtr(IdType lvid, unsigned layer);
    static FeatType *ghostVertexActivationDataPtr(IdType lvid, unsigned layer);

    static FeatType *localVertexDataBufPtr(IdType lvid, unsigned layer);
    static FeatType *localVertexLabelsPtr(IdType lvid);

    // Aggregation operation (along with normalization).
    static void aggregateFromNeighbors(IdType lvid);

    // For initialization.
    static void parseArgs(int argc, char* argv[]);
    static void readLayerConfigFile(std::string& layerConfigFileName);
    static void readFeaturesFile(std::string& featuresFileName);
    static void readLabelsFile(std::string& labelsFileName);
    static void readPartsFile(std::string& partsFileName, Graph& lGraph);
    static void processEdge(IdType& from, IdType& to, Graph& lGraph, std::set<IdType>* inTopics, std::set<IdType>* oTopics); 
    static void findGhostDegrees(std::string& fileName);
    static void setEdgeNormalizations();
    static void readGraphBS(std::string& fileName, std::set<IdType>& inTopics, std::vector<IdType>& outTopics);

    // Metric printing.
    static void printGraphMetrics();
    static void printEngineMetrics();
};


#endif //__ENGINE_HPP__
