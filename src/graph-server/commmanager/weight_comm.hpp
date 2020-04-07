#ifndef __WEIGHT_COMM_HPP__
#define __WEIGHT_COMM_HPP__

#include <vector>
#include <string>
#include <zmq.hpp>

class WeightComm {
public:
    WeightComm(std::string wserversFile, unsigned _wserverPort);
    ~WeightComm();

    // communicate with weight servers
    void updateChunkCnt(unsigned numChunks);
    void shutdown();

    unsigned wserverPort;
    unsigned wserverCnt;
    zmq::context_t ctx;
    std::vector<zmq::socket_t> wsockets;
};

void sendInfoMessage(zmq::socket_t& wsocket, unsigned cnt);

#endif // __WEIGHT_COMM_HPP__