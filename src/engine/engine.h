#ifndef __ENGINE_H__
#define __ENGINE_H__

#include "graph.hpp"
#include "../parallel/threadpool.h"
#include "bitsetscheduler.hpp"
#include "vertexprogram.hpp"
#include "../parallel/lock.hpp"
#include "../parallel/barrier.hpp"
#include "../utils/utils.h"
#include "enginecontext.h"
#include <set>
#include <vector>
#include <climits>
#include <atomic>
#include <tuple>
#include <cstdarg>

#define NUM_DATA_THREADS 1
#define NUM_COMP_THREADS 5
#define PUSHOUT_FREQUENCY 2

#define INF 1000000000  // 1B
#define MILLION 1000000 // 1M

#define NUM_DATA_THREADS_STR "1"
#define NUM_COMP_THREADS_STR "5"
#define NUM_REP_THREADS_STR "1"
#define NUM_REC_THREADS_STR "1"
#define CHKPT_FREQUENCY_STR "5"
#define PUSHOUT_FREQUENCY_STR "2"

#define ZERO 0
#define ZERO_STR "0"

#define DEFAULT_CONFIG_FILE "../config/kconf.conf"
#define HOST_FILE "../config/hostfile"
#define ZKHOST_FILE "../config/zkhostfile"
#define PARTS_EXT ".parts"
#define EDGES_EXT ".edges"
#define DELS_EXT ".dels"

// For halting decisions.
#define IAMDONE (MAX_IDTYPE - 2)
#define IAMNOTDONE (MAX_IDTYPE - 3)
#define ITHINKIAMDONE (MAX_IDTYPE - 4)

// For push out requests and responds. Seems useless!
#define PUSHOUT_REQ_END (MAX_IDTYPE - 6)
#define PUSHOUT_REQ_BEGIN (PUSHOUT_REQ_END - numNodes)
#define PUSHOUT_REQ_ME (PUSHOUT_REQ_BEGIN + nodeId)
#define PUSHOUT_RESP_END (PUSHOUT_REQ_BEGIN)
#define PUSHOUT_RESP_BEGIN (PUSHOUT_RESP_END - numNodes)
#define PUSHOUT_RESP_ME (PUSHOUT_RESP_BEGIN + nodeId)
//#define REQUEST_VERTEX (MAX_IDTYPE - 6)
//#define RESPONSE_VERTEX (MAX_IDTYPE - 7)
// -----

// Following are for control channel
#define TRY_HALT (MAX_IDTYPE - 5)
#define SNAPSHOT (MAX_IDTYPE - 6)
#define REQUEST_VERTEX_END (MAX_IDTYPE - 7)
#define RESPONSE_VERTEX_END (MAX_IDTYPE - 8)
// -----

/** Global node barriers. */
#define LAYER_BARRIER "layer"
#define RUN_BARRIER "run"



enum InOutType {SRC, DST, BOTH, NEITHER};

template <typename VertexType>
struct ReplicationType {
    IdType vId;
    VertexType vData;
    ReplicationType(IdType vid = MAX_IDTYPE, VertexType vdata = VertexType()) : vId(vid), vData(vdata) { }
};

template <typename VertexType>
struct ReplicationMType {
    IdType vId;
    VertexType vData;
    unsigned numInEdges;
    unsigned numOutEdges;
    ReplicationMType(IdType vid = MAX_IDTYPE, VertexType vdata = VertexType(), unsigned nInEdges = 0, unsigned nOutEdges = 0) : vId(vid), vData(vdata), numInEdges(nInEdges), numOutEdges(nOutEdges) { }
};

template <typename VertexType>
struct BSHeaderType {
    int sizeOfVertexType;
    VertexType numVertices;
    unsigned long long numEdges;
};

template <typename VertexType, typename EdgeType>
struct LightEdge {
    IdType fromId;
    IdType toId;

    VertexType from;
    VertexType to;
    
    EdgeType edge;
    bool valid;
};

template <typename T>
T sumReducer(T left, T right) {
  return(left += right);
}

template <typename VertexType, typename EdgeType>
class Engine {

public:

    static Graph<VertexType, EdgeType> graph;
    static BitsetScheduler* scheduler;

    static DenseBitset* shadowScheduler;
    static DenseBitset* trimScheduler;

    static unsigned dThreads;
    static ThreadPool* dataPool;

    static unsigned cThreads;
    static ThreadPool* computePool;

    static unsigned poFrequency;

    static VertexProgram<VertexType, EdgeType>* vertexProgram;
    static EdgeType (*edgeWeight) (IdType, IdType);

    static IdType currId;
    static Lock lockCurrId;

    static std::string graphFile;
    static std::string featuresFile;
    
    static unsigned nodeId;
    static unsigned numNodes;
    static bool die;

    static bool halt;

    static bool undirected;

    static bool firstIteration;
    static unsigned iteration;

    static double timeProcess;   
    static double allTimeProcess;
    static double timeInit; 

    static unsigned baseEdges;
    static unsigned numBatches;
    static unsigned batchSize;
    static unsigned deletePercent;

    static pthread_mutex_t mtxCompWaiter;
    static pthread_cond_t condCompWaiter;

    static pthread_mutex_t mtxDataWaiter;
    static pthread_cond_t condDataWaiter;

    static pthread_mutex_t lock_recvWaiters;
    static pthread_cond_t cond_recvWaiters_empty;

    static std::map<IdType, unsigned> recvWaiters;

    static pthread_barrier_t barDebug;

    static Barrier barComp;
    static Barrier barCompData;

    static std::atomic<unsigned> remPushOut;

    static unsigned long long globalInsertStreamSize;
    static std::vector<std::tuple<unsigned long long, IdType, IdType> > insertStream;

    static unsigned long long globalDeleteStreamSize;
    static std::vector<std::tuple<unsigned long long, IdType, IdType> > deleteStream;

    static EngineContext engineContext;
    static VertexType defaultVertex;
    static EdgeType defaultEdge;

    static InOutType onAdd;
    static InOutType onDelete;

    static void (*onAddHandler) (VertexType& v);
    static void (*onDeleteHandler) (VertexType& v);
    static void (*onDeleteSmartHandler) (VertexType& v, LightEdge<VertexType, EdgeType>& e);

    static void parseArgs(int argc, char* argv[]);
    static void readGraphBS(std::string& fileName, std::set<IdType>& inTopics, std::vector<IdType>& outTopics);

    /**
     * Find the in degree of the ghost vertices
     *
     * @param fileName	the filename of the graph edge list
     */
    static void findGhostDegrees(std::string& fileName);

    /**
     * Sets the normalization factors on all the edges
     */
    static void setEdgeNormalizations();


    static int readFeaturesFile(const std::string& fileName);

    static void readDeletionStream(std::string& fileName);

    static void init(int argc, char* argv[], VertexType dVertex = VertexType(), EdgeType dEdge = EdgeType(), EdgeType (*eWeight) (IdType, IdType) = NULL);
    static void destroy();

    static void run(VertexProgram<VertexType, EdgeType>* vProgram, bool printEM);

    static void processAll(VertexProgram<VertexType, EdgeType>* vProgram);

    static void printEngineMetrics();

    static void readPartsFile(std::string& partsFileName, Graph<VertexType, EdgeType>& lGraph);
    static void setRepParts();

    static void initGraph(Graph<VertexType, EdgeType>& lGraph);
    static void processEdge(IdType& from, IdType& to, Graph<VertexType, EdgeType>& lGraph, std::set<IdType>* inTopics, std::set<IdType>* oTopics, bool streaming = false); 
 
    static void replicationReceiver(unsigned tid, void* args);

    static IdType numVertices();
    static bool master(); 

private:

    static void worker(unsigned tid, void *args);
    static void dataCommunicator(unsigned tid, void *args);

    static void printLog(const char *format, ...);
};


#endif //__ENGINE_H__
