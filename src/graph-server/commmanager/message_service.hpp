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
    MessageService(){};
    MessageService(unsigned wPort_, unsigned nodeId_);

    // weight server related
    void setUpWeightSocket(char *addr);
    void prefetchWeightsMatrix(unsigned totalLayers);

    void sendWeightUpdate(Matrix &matrix, unsigned layer);
    void sendAccloss(float acc, float loss, unsigned vtcsCnt);
    // void terminateWeightServers(std::vector<char *> &weightServerAddrs);
    // void sendShutdownMessage(zmq::socket_t &weightsocket);

    Matrix getWeightMatrix(unsigned layer);

   private:
    static char weightAddr[50];
    zmq::context_t wctx;
    zmq::socket_t *dataSocket;
    zmq::socket_t *weightSocket;
    zmq::message_t confirm;
    unsigned nodeId;
    unsigned wPort;
    bool wsocktReady;

    unsigned epoch;
    std::vector<Matrix *> weights;
    std::thread wReqThread;
    std::thread wSndThread;
};

#endif
