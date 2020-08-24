#ifndef __NTWK_OPS_HPP__
#define __NTWK_OPS_HPP__

#include <unistd.h>

#include <zmq.hpp>

#include "../../../common/matrix.hpp"
#include "../../../common/utils.hpp"

#include "../utils.hpp"

#define TIMEOUT_PERIOD (2000) // ms

#define SND_MORE true
#define NO_MORE false

#define RESEND false

static inline int timedRecv(zmq::socket_t& socket, zmq::message_t& msg, unsigned ms) {
    BackoffSleeper b;
    Timer t;

    t.start();
    bool recvd = socket.recv(&msg, ZMQ_DONTWAIT);
    while (t.peek() < ms && !recvd) {
        b.sleep();
        recvd = socket.recv(&msg, ZMQ_DONTWAIT);
    }

    if (recvd) return 0;
    
    return 3;
}


int recvTensor(zmq::socket_t& socket, Matrix &mat, unsigned timeout = UINT_MAX);

void reqTensor(zmq::socket_t& socket, Chunk& chunk, std::string tensorName, bool sndMore = false);

std::vector<Matrix> reqTensors(zmq::socket_t& socket, Chunk &chunk,
                            std::vector<std::string>& tensorRequests, unsigned timeout = UINT_MAX);

Matrix reqEdgeTensor(zmq::socket_t& socket, Chunk& chunk, std::string name);

EdgeInfo reqEdgeInfo(zmq::socket_t& socket, Chunk& chunk);

int sendTensors(zmq::socket_t& socket, Chunk &chunk,
    std::vector<Matrix>& matrices, bool ack = false);

int sendEdgeTensors(zmq::socket_t& socket, Chunk& chunk,
        std::vector<Matrix>& matrices, bool ack = false);

void sendAccLoss(zmq::socket_t &dsocket, zmq::socket_t &wsocket, Matrix &predicts, Matrix &labels, Chunk &chunk);

int sendFinMsg(zmq::socket_t& socket, Chunk &chunk);

static inline void
populateHeader(void* header, unsigned op, Chunk &chunk) {
    char *ptr = (char *)header;
    memcpy(ptr, &op, sizeof(unsigned));
    memcpy(ptr + sizeof(unsigned), &chunk, sizeof(chunk));
}


#endif
