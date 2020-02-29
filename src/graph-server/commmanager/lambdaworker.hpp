#ifndef __LAMBDA_WORKER_HPP__
#define __LAMBDA_WORKER_HPP__


#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <mutex>
#include <random>
#include <sstream>
#include <climits>
#include <string>
#include <thread>
#include <vector>
#include <zmq.hpp>
#include <unistd.h>

#include "../utils/utils.hpp"
#include "../parallel/lock.hpp"
#include "../../common/matrix.hpp"
#include "../../common/utils.hpp"



class LambdaComm;

/**
 *
 * Base class for a lambda communication worker.
 *
 */
class LambdaWorker {

public:

    LambdaWorker(LambdaComm *manager_, PairQueue* _q_ptr,
      std::map<std::string, Matrix>* _savedTensors);

    ~LambdaWorker();

    // Continuously listens for incoming lambda connections.
    void work();

    // Used at context creation / destruction.
    void refreshState(Matrix actMatrix_, FeatType *zData_, FeatType *actData_,
      unsigned numFeatsNext_, bool _pipeline = false);
    void refreshState(Matrix oldGradMatrix_, Matrix newGradMatrix_,
      Matrix targetMatrix_, std::vector<Matrix> *savedTensors,
      bool _pipeline = false);

    unsigned tid;

protected:
    LambdaComm *manager;

    zmq::socket_t workersocket;

private:

    //
    // Forward-prop stuff.
    //

    // Partitions the data matrix according to the partition id and
    // send that partition to the lambda thread for computation.
    void sendAggregatedChunk(zmq::message_t& client_id, unsigned partId, unsigned layer);

    // Accepts an incoming connection from a lambda thread and receives
    // two matrices, a 'Z' matrix and a corresponding 'activations' matrix.
    void recvLambdaResults(zmq::message_t& client_id, unsigned partId, unsigned layer);
    void fakeRecvChunks(zmq::message_t& client_id, unsigned chunkCnt);

    Matrix actMatrix;   // Current layer's feats.
    unsigned numFeatsNext;
    FeatType *zData;    // Places to store the results from lambda.
    FeatType *actData;

    //
    // Backward-prop stuff.
    //
    void sendGCNChunks(zmq::message_t &client_id, unsigned partId, unsigned layer);

    // Partitions the needed matrices according to the partition id and
    // send that partition to the lambda thread for computation.
    void sendChunk(Matrix &srcMat, zmq::message_t& client_id, unsigned partId, unsigned layer, bool forward);
    void recvChunk(Matrix &dstMat, zmq::message_t& client_id, unsigned partId, unsigned layer, bool forward);

    // named-tensors
    void sendTensor(FeatType* dptr, std::string tensorName, unsigned rows,
      unsigned cols, unsigned& more);
    void getPartitionInfo(Matrix& tensor, unsigned partId, unsigned& more);
    void sendTensors(unsigned partId, zmq::message_t& client_id);

    Matrix recvTensor();
    void recvTensors(unsigned partId, zmq::message_t& client_id);
    // end named-tensors

    // Partitions the label matrix given a partition id and
    // and send that partition to the lambda thread for validation
    void sendTargetMatrix(zmq::message_t& client_id, unsigned partId);

    // Receive the summed loss and total correct for this model
    void recvValidationResults(zmq::message_t& client_id, zmq::message_t& header);

    Matrix oldGradMatrix;
    Matrix newGradMatrix;
    Matrix targetMatrix;
    std::vector<Matrix> *savedTensors;
    std::map<std::string, Matrix>* savedVtxTensors;

    // Callback when lambda results are returned
    bool pipeline;
    PairQueue* q_ptr;

    // timing info for profiling
    unsigned recvTS;
    unsigned sendTS;

    void setTSinHdr(void *hdr_buf) {
        sendTS = timestamp_ms();
        *((unsigned *)hdr_buf + 5) = recvTS;
        *((unsigned *)hdr_buf + 6) = sendTS;
    }
    void setTSinCfm(void *cfm_buf) {
        sendTS = timestamp_ms();
        *((unsigned *)cfm_buf + 1) = recvTS;
        *((unsigned *)cfm_buf + 2) = sendTS;
    }
};


#endif
