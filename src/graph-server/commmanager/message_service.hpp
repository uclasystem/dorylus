#ifndef __MSG_SRV_HPP__
#define __MSG_SRV_HPP__

#include <chrono>
#include <thread>
#include <vector>
#include <zmq.hpp>

#include "../../common/matrix.hpp"
#include "../../common/utils.hpp"
#include "../utils/utils.hpp"

// This class is used for CPU/GPU <-> weight server communication
class MessageService {
public:
    MessageService(unsigned wPort_, unsigned nodeId_,
                   unsigned numLayers_, GNN gnn_type);

    // weight server related
    void setUpWeightSocket(char *addr);
    void prefetchWeightsMatrix();

    // for 'w' weight matrix
    Matrix getWeightMatrix(unsigned layer);
    void sendWeightUpdate(Matrix &matrix, unsigned layer);
    // for 'a_i' weight matrix
    Matrix getaMatrix(unsigned layer);
    void sendaUpdate(Matrix &matrix, unsigned layer);

    void sendAccloss(float acc, float loss, unsigned vtcsCnt);

private:
    zmq::context_t wctx;
    zmq::socket_t wsocket;
    zmq::message_t confirm;
    unsigned nodeId;
    unsigned wPort;
    bool wsocktReady;

    GNN gnn_type;
    unsigned epoch;
    unsigned numLayers;

    std::vector<Matrix> weights;
    std::vector<Matrix> as;
    std::thread wReqThread;
    std::thread wSndThread;
};

#endif
