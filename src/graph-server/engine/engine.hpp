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
#include "../parallel/lock.hpp"
#include "../parallel/cond.hpp"
#include "../utils/utils.hpp"
#include "../../common/matrix.hpp"

// Max size (bytes) for a message received by the data communicator.
#define MAX_MSG_SIZE (1 * 1024 * 1024)
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

class LockChunkQueue {
public:
    void lock() { lk.lock(); }
    void unlock() { lk.unlock(); }

    bool empty() const { return cq.empty(); }
    size_t size() const { return cq.size(); }
    const Chunk &top() const { return cq.top(); }
    void push(const Chunk &chunk) { cq.push(chunk); }
    void push_atomic(const Chunk &chunk) {
        lk.lock();
        cq.push(chunk);
        lk.unlock();
    }
    void pop() { cq.pop(); }
    void clear() {
        while (!cq.empty())
            cq.pop();
    }
private:
    Lock lk;
    ChunkQueue cq;
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
    void preallocate_tensors(GNN gnn_type);
    void preallocateGCN();
    void preallocateGAT();

    void run();
    void runPipeline();
    void runAsyncPipelineGCN(); // deprecated

    void output();
    void destroy();
    bool master();

    // HIGH LEVEL SAGA FUNCITONS
    void aggregateGCN(Chunk &chunk);
    void applyVertexGCN(Chunk &chunk);
    void scatterGCN(Chunk &chunk);
    void applyEdgeGCN(Chunk &chunk);

    void aggregateGAT(Chunk &chunk);
    void predictGAT(Chunk &chunk);
    void applyVertexGAT(Chunk &chunk);
    void scatterGAT(Chunk &chunk);
    void applyEdgeGAT(Chunk &chunk);
    LockChunkQueue schQueue;
    LockChunkQueue GAQueue;
    LockChunkQueue AVQueue;
    LockChunkQueue SCQueue;
    LockChunkQueue AEQueue;
    void gatherWorkFunc(unsigned tid);
    void applyVertexWorkFunc(unsigned tid);
    void scatterWorkFunc(unsigned tid);
    void ghostReceiverFunc(unsigned tid);
    void ghostReceiverGCN(unsigned tid);
    void ghostReceiverGAT(unsigned tid);
    void applyEdgeWorkFunc(unsigned tid);
    void scheduleFunc(unsigned tid);
    void scheduleAsyncFunc(unsigned tid);
    LockChunkQueue SCStashQueue;
    PROP_TYPE currDir;
    bool pipelineHalt = false;
    bool async = false;

    unsigned getAbsLayer(const Chunk &c);
    Chunk incLayerGCN(const Chunk &c);
    Chunk incLayerGAT(const Chunk &c);
    bool isLastLayer(const Chunk &c);

    void loadChunks();

    // TENSOR OPS
    // NOTE: Implementing in engine for now but need to move later
    FeatType* softmax(FeatType* inputTensor, FeatType* result, unsigned rows, unsigned cols);
    Matrix softmax_prime(FeatType* valuesTensor, FeatType* softmaxOutput, unsigned size);
    Matrix sparse_dense_elemtwise_mult(CSCMatrix<EdgeType>& csc,
      FeatType* sparseInputTensor, FeatType* denseInputTensor);
    FeatType* leakyReLU(FeatType* matxData, unsigned vecSize);
    FeatType leakyReLU(FeatType f);

    void makeBarrier();

    unsigned getNumEpochs();
    unsigned getValFreq();
    unsigned getNodeId();

    void addEpochTime(double epochTime);

// private:
    NodeManager nodeManager;
    CommManager commManager;

    Graph graph;

    unsigned dThreads;
    unsigned cThreads;

    GNN gnn_type;

    // Config of number of features in each layer.
    std::vector<unsigned> layerConfig;
    unsigned numLayers = 0;

    // intermediate data for vertex/edge NN backward computation.
    std::vector<Matrix> *vtxNNSavedTensors;
    std::vector<Matrix> *edgNNSavedTensors;

    std::vector< TensorMap > savedNNTensors;
    std::vector< ETensorMap > savedEdgeTensors;

    // Persistent pointers to original input data
    FeatType *forwardVerticesInitData;
    FeatType *forwardGhostInitData;
    // Labels one-hot storage array.
    FeatType *localVerticesLabels = NULL;

    // For pipeline scatter sync
    int recvCnt = 0;
    Lock recvCntLock;
    Cond recvCntCond;
    int ghostVtcsRecvd;

    // Read-in files
    std::string datasetDir;
    std::string outFile;
    std::string featuresFile;
    std::string layerConfigFile;
    std::string labelsFile;
    std::string dshMachinesFile;
    std::string myPrIpFile;
    std::string myPubIpFile;

    bool forcePreprocess = false;

    std::time_t start_time;
    std::time_t end_time;

    unsigned dataserverPort;
    unsigned weightserverPort;
    std::string weightserverIPFile;

    std::string lambdaName;
    unsigned numLambdasForward = 0;
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

    bool undirected = false;

    unsigned layer = 0;
    const unsigned START_EPOCH = 0;
    unsigned currEpoch = START_EPOCH;

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

    void calcAcc(FeatType *predicts, FeatType *labels, unsigned vtcsCnt,
                 unsigned featDim);

    // Worker and communicator thread function.
    void verticesPushOut(unsigned receiver, unsigned totCnt, unsigned *lvids,
      FeatType *inputTensor, unsigned featDim, Chunk& c);
    void sendEpochUpdate(unsigned currEpoch);

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

    unsigned timeoutRatio;
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

extern Engine engine;

#endif //__ENGINE_HPP__
