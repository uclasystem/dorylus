#include <cstdio>
#include <fstream>
#include <cassert>
#include "../utils/utils.hpp"
#include "weight_comm.hpp"
#include <boost/algorithm/string/trim.hpp>

WeightComm::WeightComm(std::string wserversFile, unsigned _wserverPort) :
    wserverPort(_wserverPort), ctx(1) {
    std::vector<char*> wserverAddrs;

    std::ifstream infile(wserversFile);
    if (!infile.good())
        fprintf(stderr, "Cannot open weight server file: %s [Reason: %s]\n",
                 wserversFile.c_str(), std::strerror(errno));
    assert(infile.good());

    std::string line;
    while (!infile.eof()) {
        std::getline(infile, line);
        boost::algorithm::trim(line);

        if (line.length() == 0)
            continue;
        char *addr = strdup(line.c_str());

        // While reading in string, also initialize connection if master
        unsigned ind = wserverAddrs.size();
        wsockets.push_back(zmq::socket_t(ctx, ZMQ_DEALER));
        char identity[] = "graph-master";
        wsockets[ind].setsockopt(ZMQ_IDENTITY, identity, strlen(identity) + 1);
        char whost_port[50];
        sprintf(whost_port, "tcp://%s:%u", addr, wserverPort);
        wsockets[ind].connect(whost_port);

        wserverAddrs.push_back(addr);
        free(addr);
    }
    wserverCnt = wsockets.size();
}

WeightComm::~WeightComm() {
    for (zmq::socket_t &wsocket : wsockets) {
        wsocket.setsockopt(ZMQ_LINGER, 0);
        wsocket.close();
    }
    while (!wsockets.empty()) {
        wsockets.pop_back();
    }
    ctx.close();
}

void WeightComm::updateChunkCnt(unsigned chunkCnt) {
    unsigned base = chunkCnt / wsockets.size();
    unsigned remainder = chunkCnt % wsockets.size();

    for (unsigned u = 0; u < remainder; ++u) {
        sendInfoMessage(wsockets[u % wsockets.size()], base + 1);
    }

    for (unsigned u = remainder; u < wsockets.size(); ++u) {
        sendInfoMessage(wsockets[u % wsockets.size()], base);
    }
}

void WeightComm::shutdown() {
    fprintf(stderr, "Terminating weight servers\n");

    for (zmq::socket_t& wsocket : wsockets) {
        zmq::message_t header(HEADER_SIZE);
        populateHeader((char*) header.data(), OP::TERM);
        wsocket.send(header);

        // zmq::message_t ack;
        // wsocket.recv(&ack);
    }
}

void sendInfoMessage(zmq::socket_t& wsocket, unsigned cnt) {
    zmq::message_t info_header(HEADER_SIZE);
    populateHeader((char*) info_header.data(), OP::INFO, cnt);
    wsocket.send(info_header);

    zmq::message_t ack;
    wsocket.recv(&ack);
}