#ifndef __ENGINE_HPP__
#define __ENGINE_HPP__

#include <set>
#include <vector>
#include <climits>
#include <atomic>
#include <tuple>
#include <cstdio>
#include <mutex>
#include <condition_variable>

#include "../graph/graph.hpp"
#include "../commmanager/commmanager.hpp"
#include "../commmanager/weight_comm.hpp"
#include "../commmanager/resource_comm.hpp"
#include "../nodemanager/nodemanager.hpp"
#include "../parallel/threadpool.hpp"
#include "../parallel/lock.hpp"
#include "../parallel/cond.hpp"
#include "../parallel/barrier.hpp"
#include "../utils/utils.hpp"
#include "../../common/matrix.hpp"

// Max size (bytes) for a message received by the data communicator.
#define MAX_MSG_SIZE (5 * 1024 * 1024)
#define NODE_ID_DIGITS 8 // Digits num of node id.
#define NODE_ID_HEADER "%8X" // Header for node id. For communication.
#define DATA_HEADER_SIZE (NODE_ID_DIGITS + sizeof(unsigned) * 5)



/** Binary features file header struct. */
struct FeaturesHeaderType {
    unsigned numFeatures;
};

/** Binary labels file header struct. */
struct LabelsHeaderType {
    unsigned labelKinds;
};


enum GNN { GCN };


/**
 *
 * Class of a GNN-LAMBDA engine executing on a node.
 *
 */
class Engine {
public:
    // Public APIs for benchmarks.
    void init(int argc, char *argv[]);
    void preallocate_tensors(GNN gnn_type);
    void preallocateGCN();

    FeatType *runForward(unsigned epoch);
    void runBackward(FeatType *backwardInitData);

    void run();
    void runAsyncPipeline();

    void runForwardSyncPipeline(unsigned epoch);
    void runBackwardSyncPipline();

    void output();
    void destroy();
    bool master();

    // HIGH LEVEL SAGA FUNCITONS
    FeatType* aggregate(FeatType **eVFeatsTensor, unsigned edgsCnt,
                        unsigned featDim, AGGREGATOR aggregator);
    FeatType* applyVertex(FeatType *vtcsTensor, unsigned vtcsCnt,
                        unsigned inFeatDim, unsigned outFeatDim, bool lastLayer);
    FeatType** scatter(FeatType *vtcsTensor, unsigned vtcsCnt, unsigned featDim);
    FeatType** applyEdge(EdgeType *edgsTensor, unsigned edgsCnt, unsigned eFeatDim,
                        FeatType **eSrcVFeatsTensor, FeatType **eDstVFeatsTensor,
                        unsigned inFeatDim, unsigned outFeatDim);

    FeatType* fusedGatherApply(FeatType **eVFeatsTensor, unsigned edgsCnt,
                        unsigned inFeatDim, unsigned outFeatDim, AGGREGATOR aggregator);
    void fusedGAS();

    FeatType* aggregateBackward(FeatType **eVFeatsTensor, unsigned edgsCnt,
                        unsigned featDim, AGGREGATOR aggregator);
    FeatType* applyVertexBackward(FeatType *gradTensor, unsigned vtcsCnt,
                        unsigned inFeatDim, unsigned outFeatDim);
    FeatType** scatterBackward(FeatType *gradTensor, unsigned vtcsCnt,
                               unsigned featDim);
    FeatType** applyEdgeBackward(EdgeType *edgsTensor, unsigned edgsCnt, unsigned eFeatDim,
                        FeatType **eSrcVGradTensor, FeatType **eDstVGradTensor,
                        unsigned inFeatDim, unsigned outFeatDim);

    FeatType* fusedGatherApplyBackward(FeatType **eVFeatsTensor, unsigned edgsCnt,
                        unsigned inFeatDim, unsigned outFeatDim, AGGREGATOR aggregator);
    FeatType* fusedGASBackward(FeatType* gradTensor, unsigned vtcsCnt,
                        unsigned inFeatDim, unsigned outFeatDim,
                        bool aggregate, bool scatter);

    void makeBarrier();

    unsigned getNumEpochs();
    unsigned getValFreq();
    unsigned getNodeId();

    void setPipeline(bool _pipeline);
    void addEpochTime(double epochTime);

// private:
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

    std::vector< TensorMap > savedNNTensors;
    std::vector< ETensorMap > savedEdgeTensors;

    // intermediate data for backward computation.
    std::vector<Matrix> *savedTensors;
    TensorMap savedVtxTensors;

    // Persistent pointers to original input data
    FeatType *forwardVerticesInitData;
    FeatType *forwardGhostInitData;

    FeatType *forwardGhostVerticesDataIn;
    FeatType *forwardGhostVerticesDataOut;
    FeatType *backwardGhostVerticesDataIn;
    FeatType *backwardGhostVerticesDataOut;

    struct AggOPArgs {
        FeatType *outputTensor;
        FeatType **inputTensor;
        unsigned vtcsCnt;
        unsigned edgsCnt;
        unsigned featDim;
    };

    // Labels one-hot storage array.
    FeatType *localVerticesLabels = NULL;

    // YIFAN: this is used together with edgsTensor. edgsTensor are arrays of pointers to this underlying vtcs tensors buf.
    // TODO: (YIFAN) This is not elegant. Think a way to get rid of this.
    FeatType *underlyingVtcsTensorBuf;

    unsigned currId = 0;

    int recvCnt = 0;
    Lock recvCntLock;
    Cond recvCntCond;
    // YIFAN: for now this is for pipeline GAS only
    int ghostVtcsRecvd;

    std::string datasetDir;
    std::string outFile;
    std::string featuresFile;
    std::string layerConfigFile;
    std::string labelsFile;
    std::string dshMachinesFile;
    std::string myPrIpFile;
    std::string myPubIpFile;

    std::time_t start_time;
    std::time_t end_time;

    unsigned dataserverPort;
    unsigned weightserverPort;
    std::string weightserverIPFile;

    unsigned numLambdasForward = 0;
    unsigned numLambdasBackward = 0;
    unsigned numEpochs = 0;
    unsigned numSyncEpochs = 0;
    unsigned numAsyncEpochs = 0;
    unsigned valFreq = 0;

    float accuracy = 0.0;

    //0: Lambda, 1: GPU, 2: CPU
    unsigned mode = 0;

    WeightComm *weightComm;
    ResourceComm *resComm;
    unsigned nodeId;
    unsigned numNodes;

    bool commHalt = false;
    bool aggHalt = false;

    bool undirected = false;

    unsigned layer = 0;
    unsigned currEpoch = -1u;

    // Timing stuff.
    double timeInit = 0.0;
    double timeForwardProcess = 0.0;
    double timeBackwardProcess = 0.0;
    std::vector<double> vecTimeAggregate;
    std::vector<double> vecTimeApplyVtx;
    std::vector<double> vecTimeLambdaInvoke;
    std::vector<double> vecTimeLambdaWait;
    std::vector<double> vecTimeApplyEdg;
    std::vector<double> vecTimeScatter;
    std::vector<double> epochTimes;
    double asyncAvgEpochTime;

    std::vector<unsigned> epochMs;

    Barrier barComp;


    void calcAcc(FeatType *predicts, FeatType *labels, unsigned vtcsCnt,
                 unsigned featDim);

    // Worker and communicator thread function.
    void forwardGhostReceiver(unsigned tid);
    void backwardGhostReceiver(unsigned tid);

    void aggregator(unsigned tid);
    void scatterWorker(unsigned tid);
    void ghostReceiver(unsigned tid);

    void gasAgger(unsigned tid);
    void gasScter(unsigned tid);
    void gasRcver(unsigned tid);


    void verticesPushOut(unsigned receiver, unsigned totCnt, unsigned *lvids,
      FeatType *inputTensor, unsigned featDim, Chunk& c);
    void sendEpochUpdate(unsigned currEpoch);

    void aggregateCompute(unsigned tid, void *args);
    void aggregateBPCompute(unsigned tid, void *args);

    void aggregateChunk(Chunk& c);
    void aggregateBPChunk(Chunk& c);

    void gatherApplyCompute(unsigned tid, void *args);
    void gatherApplyBPCompute(unsigned tid, void *args);

    // transform from vtxFeats/edgFeats to edgFeats/vtxFeats
    FeatType** srcVFeats2eFeats(FeatType *vtcsTensor, FeatType* ghostTensor, unsigned vtcsCnt, unsigned featDim);
    FeatType** dstVFeats2eFeats(FeatType *vtcsTensor, FeatType* ghostTensor, unsigned vtcsCnt, unsigned featDim);
    // FeatType* eFeats2dstVFeats(FeatType **edgsTensor, unsigned edgsCnt, unsigned featDim);
    // FeatType* eFeats2srcVFeats(FeatType **edgsTensor, unsigned edgsCnt, unsigned featDim);

    // About the global data arrays.
    inline unsigned getFeatDim(unsigned layer) {
        return layerConfig[layer];
    }

    inline FeatType *localVertexLabelsPtr(unsigned lvid) {
        return localVerticesLabels + lvid * getFeatDim(numLayers);
    }

    void sendForwardGhostUpdates(FeatType *inputTensor, unsigned featDim);
    void sendBackwardGhostGradients(FeatType *gradTensor, unsigned featDim);

    // All pipeline related functions/members
    void loadChunks();

    void pipelineForwardGhostUpdates(unsigned tid);
    void pipelineBackwardGhostGradients(FeatType* inputTensor, unsigned featDim);

    void pipelineGhostReceiver(unsigned tid);

    // fusedGASBackward phases
    FeatType* applyScatterPhase(FeatType* gradTensor, unsigned vtcsCnt,
      unsigned inFeatDim, unsigned outFeatDim, bool scatter);
    FeatType* aggregateApplyScatterPhase(FeatType* gradTensor, unsigned vtcsCnt,
      unsigned inFeatDim, unsigned outFeatDim, bool scatter);
    FeatType* aggregateApplyPhase(FeatType* gradTensor, unsigned vtcsCnt,
      unsigned inFeatDim, unsigned outFeatDim, bool scatter);

    ChunkQueue aggregateQueue;
    Lock aggQueueLock;
    ChunkQueue scatterQueue;
    Lock scatQueueLock;

    unsigned staleness;
    volatile CONVERGE_STATE convergeState = CONVERGE_STATE::EARLY;
    unsigned minEpoch;
    unsigned maxEpoch;
    // Tracking how many chunks have finished each epoch in the
    // local partition
    std::vector<unsigned> numFinishedEpoch;
    Lock finishedChunkLock;
    // Tracking how many graph servers have finished a certain epoch
    // Works just like numFinishedEpoch but with nodes not chunks
    std::vector<unsigned> nodesFinishedEpoch;
    Lock finishedNodeLock;
    unsigned finishedChunks;

    bool pipeline = false;
    // END Pipeline related functions/members

    // Ghost update operation, send vertices to other nodes
    void forwardVerticesPushOut(unsigned receiver, unsigned totCnt,
                                unsigned *lvids, FeatType *inputTensor,
                                unsigned featDim);
    void backwardVerticesPushOut(unsigned receiver, unsigned totCnt,
                                 unsigned *lvids, FeatType *gradTensor,
                                 unsigned featDim);

    // Both data push out funcs have same logic. Plan to replace
    // them with this one
    void verticesDataPushOut(unsigned receiver, unsigned totCtn,
                              unsigned *lvids, FeatType* tensor,
                              unsigned featDim);

    // Aggregation operation (along with normalization).
    void forwardAggregateFromNeighbors(unsigned lvid, FeatType *outputTensor,
                                    FeatType **inputTensor, unsigned featDim);
    void backwardAggregateFromNeighbors(unsigned lvid, FeatType *nextGradTensor,
                                    FeatType **gradTensor, unsigned featDim);

    void saveTensor(std::string& name, unsigned rows, unsigned cols, FeatType *dptr);
    void saveTensor(const char* name, unsigned rows, unsigned cols, FeatType *dptr);
    void saveTensor(Matrix& mat);

    void saveTensor(const char* name, unsigned layer, unsigned rows, unsigned cols, FeatType *dptr);
    void saveTensor(const char* name, unsigned layer, Matrix& mat);

    // For initialization.
    void parseArgs(int argc, char* argv[]);
    void readLayerConfigFile(std::string& layerConfigFileName);
    void readFeaturesFile(std::string& featuresFileName);
    void readLabelsFile(std::string& labelsFileName);

    // Metric printing.
    void printGraphMetrics();
    void printEngineMetrics();
};

// Fetch vertex feature from vtx feats array
#define getVtxFeat(dataBuf, lvid, featDim) ((dataBuf) + (lvid) * (featDim))

// Every one includes this file can access the static engine object now
extern Engine engine;

#endif //__ENGINE_HPP__
