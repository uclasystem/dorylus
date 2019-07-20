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

#define DATACOMM_BARRIER "datacomm"
#define COMM_BARRIER "comm"



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

    static bool compDone;
    static bool halt;

    static bool undirected;

    static bool firstIteration;
    static unsigned iteration;

    static double timProcess;   
    static double allTimProcess;
    static double timInit; 

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

    /*
        Usage:This function will read feature file and 
            set data to feature vectors of both local and 
            ghost vertices.
        The function is called in engine::init()
            The fileName is currently hard coded to "../inputs/features.txt"
        The file being read should have format of(either "," or " " is fine):
        f00,f01,f02,....,f0n
        f10,f11,f12,....,f1n
        ...
        fm0,fm1,fm2,....,fmn
        

        It is also worth noting that since ghost vertex doesn't have attribute Id,
        it would be easier for you to test by setting the first feature 
        in the feature vector as nodeId.
    */
    static int readFeaturesFile(const std::string& fileName);

    static void readDeletionStream(std::string& fileName);
    static void addEdge(IdType from, IdType to, std::set<IdType>* inTopics);
    static void deleteEdge(IdType from, IdType to);
    static LightEdge<VertexType, EdgeType> deleteEdge2(IdType from, IdType to);
    static bool deleteEdge3(IdType from, IdType to);

    static void receiveNewGhostValues(std::set<IdType>& inTopics);

    static void init(int argc, char* argv[], VertexType dVertex = VertexType(), EdgeType dEdge = EdgeType(), EdgeType (*eWeight) (IdType, IdType) = NULL);
    static void destroy();
    
    static void setOnAddDelete(InOutType oa, void (*oaHandler) (VertexType& v), InOutType od, void (*odHandler) (VertexType& v));
    static void setOnDeleteSmartHandler(void (*odSmartHandler) (VertexType& v, LightEdge<VertexType, EdgeType>& e));

    static void run(VertexProgram<VertexType, EdgeType>* vProgram, bool printEM);
    static void streamRun(VertexProgram<VertexType, EdgeType>* vProgram, VertexProgram<VertexType, EdgeType>* wProgram, void (*reset)(void), bool printEM);
    static void streamRun2(VertexProgram<VertexType, EdgeType>* vProgram, VertexProgram<VertexType, EdgeType>* removeApproximations, VertexProgram<VertexType, EdgeType>* exactProgram, VertexProgram<VertexType, EdgeType>* wProgram, bool smartDeletions, bool printEM);
    static void streamRun3(VertexProgram<VertexType, EdgeType>* vProgram, VertexProgram<VertexType, EdgeType>* tagProgram, VertexProgram<VertexType, EdgeType>* removeApproximations, VertexProgram<VertexType, EdgeType>* exactProgram, VertexProgram<VertexType, EdgeType>* wProgram, bool smartDeletions, bool printEM);
    static void streamRun4(VertexProgram<VertexType, EdgeType>* vProgram, VertexProgram<VertexType, EdgeType>* trimProgram, VertexProgram<VertexType, EdgeType>* wProgram, bool smartDeletions, bool printEM);

    static void streamRun5(VertexProgram<VertexType, EdgeType>* vProgram, VertexProgram<VertexType, EdgeType>* trimProgram, VertexProgram<VertexType, EdgeType>* wProgram, bool printEM);

    static void processAll(VertexProgram<VertexType, EdgeType>* vProgram);
    static void quickRun(VertexProgram<VertexType, EdgeType>* vProgram, bool metrics = false);

    static void printEngineMetrics();

    static void readPartsFile(std::string& partsFileName, Graph<VertexType, EdgeType>& lGraph);
    static void setRepParts();

    static void initGraph(Graph<VertexType, EdgeType>& lGraph);
    static void processEdge(IdType& from, IdType& to, Graph<VertexType, EdgeType>& lGraph, std::set<IdType>* inTopics, std::set<IdType>* oTopics, bool streaming = false); 
 
    static void worker(unsigned tid, void* args);

    static bool reachOutputLayer();
    static void gotoNextLayer();

    static void dataCommunicator(unsigned tid, void* args);
    static void replicationReceiver(unsigned tid, void* args);

    static void signalAll();
    static void signalVertex(IdType vId);
    
    static void shadowSignalVertex(IdType vId);
    static void shadowUnsignalVertex(IdType vId);
   
    static void trimmed(IdType vId);
    static void notTrimmed(IdType vId);
    
    static void activateEndPoints(IdType from, IdType to, InOutType io, void (*oadHandler) (VertexType& v)); 
    static void activateEndPoints2(IdType from, IdType to, InOutType io, void (*oadHandler) (VertexType& v, LightEdge<VertexType, EdgeType>& e), LightEdge<VertexType, EdgeType>& edge);

    static void updateGhostVertex(IdType vid, VertexType& value);
    static void conditionalUpdateGhostVertex(IdType vid, VertexType& value);

    static unsigned getPreviousAliveNodeId(unsigned nId);
    static unsigned getNextAliveNodeId(unsigned nId);

    static IdType numVertices();
    static bool master(); 
    
    template<typename T>
    static T sillyReduce(T value, T (*reducer)(T, T)); 
};

#endif //__ENGINE_H__
