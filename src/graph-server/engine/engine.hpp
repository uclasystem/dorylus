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


#define NUM_DATA_THREADS 1          // These are default values (when cli argument is empty).
#define NUM_COMP_THREADS 7
#define NUM_DATA_THREADS_STR "1"
#define NUM_COMP_THREADS_STR "5"


#define ZERO 0
#define ZERO_STR "0"
#define INF 1000000000  // 1B
#define MILLION 1000000 // 1M


#define MAX_MSG_SIZE 8192   // Max size (bytes) for a message received by the data communicator.


/** For files cli options. */
#define DEFAULT_CONFIG_FILE "../config/kconf.conf"
#define HOST_FILE "../config/hostfile"
#define ZKHOST_FILE "../config/zkhostfile"
#define PARTS_EXT ".parts"
#define EDGES_EXT ".edges"


/** Global node barriers. */
#define INIT_BARRIER "init"
#define RUN_BARRIER "run"
#define LAYER_BARRIER "layer"


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

    static unsigned dataserverPort;
    static std::string coordserverIp;
    static unsigned coordserverPort;

    static unsigned nodeId;
    static unsigned numNodes;

    static bool halt;

    static bool undirected;

    static unsigned iteration;

    static double timeInit;
    static double timeProcess;   

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
