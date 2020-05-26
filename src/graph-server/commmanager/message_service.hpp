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
    MessageService(unsigned wPort_, unsigned nodeId_);

    // weight server related
    void setUpWeightSocket(char *addr);

    Matrix getWeightMatrix(unsigned layer);
    Matrix getaMatrix(unsigned layer);
    void sendWeightUpdate(Matrix &matrix, unsigned layer);
    void sendaUpdate(Matrix &a, unsigned layer);

    void prefetchWeightsMatrix(unsigned totalLayers);

   private:
    static char weightAddr[50];
    zmq::context_t wctx;
    zmq::socket_t wsocket;
    zmq::message_t confirm;
    unsigned nodeId;
    unsigned wPort;
    bool wsocktReady;

    unsigned epoch;
    std::vector<Matrix> weights;
    std::vector<Matrix> as; // a vector for edge attention computation
    std::thread wReqThread;
    std::thread wSndThread;
};

#endif
