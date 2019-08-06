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
    unsigned int numFeatures;
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
    static std::vector<unsigned> layerConfigPrefixSum;   // Prefix sum of layerConfig.

    static unsigned numFeatsTotal;
    static unsigned numLayers;

    static FeatType *verticesZData;     // Global contiguous array for all vertices' data. Stored in row-wise order:
                                        // The first bunch of values are data for the 0th vertex, ...
    static FeatType *verticesActivationData;
    static FeatType *ghostVerticesActivationData;
    static FeatType *verticesDataBuf;   // A smaller buffer storing current iter's data after aggregation. (Serves as the
                                        // serialization area naturally.)

    static IdType currId;
    static Lock lockCurrId;

    static Lock lockRecvWaiters;
    static Cond condRecvWaitersEmpty;

    static Lock lockHalt;

    static std::string graphFile;
    static std::string featuresFile;
    static std::string outFile;
    static std::string layerConfigFile;
    static std::string dshMachinesFile;

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
    static std::vector<double> vecTimeWriteback;

    static std::map<IdType, unsigned> recvWaiters;

    static Barrier barComp;

    // Worker and communicator thread function.
    static void worker(unsigned tid, void *args);
    static void dataCommunicator(unsigned tid, void *args);

    // About the global data arrays.
    static unsigned getNumFeats();
    static unsigned getNumFeats(unsigned iter);

    static unsigned getDataAllOffset();
    static unsigned getDataAllOffset(unsigned iter);

    static FeatType *vertexZDataPtr(IdType lvid, unsigned offset);
    static FeatType *vertexActivationDataPtr(IdType lvid, unsigned offset);

    static FeatType *ghostVertexActivationDataPtr(IdType lvid, unsigned offset);

    static FeatType *vertexDataBufPtr(IdType lvid, unsigned numFeats);

    // Aggregation operation (along with normalization).
    static void aggregateFromNeighbors(IdType lvid);

    // For initialization.
    static void parseArgs(int argc, char* argv[]);
    static void readLayerConfigFile(std::string& layerConfigFileName);
    static void readFeaturesFile(std::string& featuresFileName);
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
