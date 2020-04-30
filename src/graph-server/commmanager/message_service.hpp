#ifndef __MSG_SRV_HPP__
#define __MSG_SRV_HPP__

#include <zmq.hpp>
#include "../utils/utils.hpp"
#include "../../common/matrix.hpp"
#include "../../common/utils.hpp"
#include <thread>
#include <vector>
#include <chrono>

//This class is used for CPU/GPU <-> weight server communication
class MessageService {
  public:
    MessageService() {};
    MessageService(unsigned wPort_, unsigned nodeId_);

    //weight server related
    void setUpWeightSocket(char *addr);
    void prefetchWeightsMatrix(unsigned totalLayers);
    void sendInfoMessage(unsigned numLambdas);

    void sendWeightUpdate(Matrix &matrix, unsigned layer);
    void terminateWeightServers(std::vector<char *> &weightServerAddrs);
    void sendShutdownMessage(zmq::socket_t &weightsocket);

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

    std::vector<Matrix *> weights;
    std::thread wReqThread;
    std::thread wSndThread;
    std::thread infoThread;
};

#endif
