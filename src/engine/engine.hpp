#ifndef __ENGINE_HPP__
#define __ENGINE_HPP__


#include <set>
#include <vector>
#include <climits>
#include <atomic>
#include <tuple>
#include "graph.hpp"
#include "vertexprogram.hpp"
#include "../commmanager/commmanager.hpp"
#include "../nodemanager/nodemanager.hpp"
#include "../parallel/threadpool.hpp"
#include "../parallel/lock.hpp"
#include "../parallel/cond.hpp"
#include "../parallel/barrier.hpp"
#include "../utils/utils.hpp"


// TODO: Hardcoded # of layers, should be changed sooner or later.
#define NUM_LAYERS 5


#define NUM_DATA_THREADS 1
#define NUM_COMP_THREADS 5
#define NUM_DATA_THREADS_STR "1"
#define NUM_COMP_THREADS_STR "5"


#define ZERO 0
#define ZERO_STR "0"
#define INF 1000000000  // 1B
#define MILLION 1000000 // 1M


/** Input files. */
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
template <typename VertexType>
struct BSHeaderType {
    int sizeOfVertexType;
    VertexType numVertices;
    unsigned long long numEdges;
};


/**
 *
 * Class of an ASPIRE engine executing on a node.
 * 
 */
template <typename VertexType, typename EdgeType>
class Engine {

private:

    static Graph<VertexType, EdgeType> graph;

    static unsigned dThreads;
    static ThreadPool *dataPool;

    static unsigned cThreads;
    static ThreadPool *computePool;

    static VertexProgram<VertexType, EdgeType> *vertexProgram;
    static EdgeType (*edgeWeight) (IdType, IdType);

    static IdType currId;
    static Lock lockCurrId;

    static Lock lockRecvWaiters;
    static Cond condRecvWaitersEmpty;

    static Lock lockHalt;

    static std::string graphFile;
    static std::string featuresFile;
    
    static unsigned nodeId;
    static unsigned numNodes;

    static bool halt;

    static bool undirected;

    static unsigned iteration;

    static double timeProcess;   
    static double allTimeProcess;
    static double timeInit;

    static std::map<IdType, unsigned> recvWaiters;

    static Barrier barComp;

    static VertexType defaultVertex;
    static EdgeType defaultEdge;

    // Worker and communicator thread function.
    static void worker(unsigned tid, void *args);
    static void dataCommunicator(unsigned tid, void *args);

    // For initialization.
    static void parseArgs(int argc, char* argv[]);
    static void readFeaturesFile(std::string& fileName);
    static void readPartsFile(std::string& partsFileName, Graph<VertexType, EdgeType>& lGraph);
    static void processEdge(IdType& from, IdType& to, Graph<VertexType, EdgeType>& lGraph, std::set<IdType>* inTopics, std::set<IdType>* oTopics); 
    static void findGhostDegrees(std::string& fileName);
    static void setEdgeNormalizations();
    static void readGraphBS(std::string& fileName, std::set<IdType>& inTopics, std::vector<IdType>& outTopics);

    // Metric printing.
    static void printEngineMetrics();

public:

    // Public APIs for benchmarks.
    static void init(int argc, char *argv[], VertexType dVertex = VertexType(), EdgeType dEdge = EdgeType(), EdgeType (*eWeight) (IdType, IdType) = NULL);
    static void run(VertexProgram<VertexType, EdgeType> *vProgram, bool printEM);
    static void processAll(VertexProgram<VertexType, EdgeType> *vProgram);
    static void destroy();
    static bool master();
};


#endif //__ENGINE_HPP__
