#ifndef __COMP_SERVER_HPP__
#define __COMP_SERVER_HPP__

#include <zmq.hpp>
#include <chrono>
#include <iostream>
#include <fstream>
#include "../comp_unit/comp_unit.hpp"
#include <string>
#include <cstring>
#include <sstream>
#include "../../../src/utils/utils.hpp"
#include <boost/algorithm/string/trim.hpp>
#include <vector>
#include <thread>

void doNotFreeBuffer(void *data, void *hint){
    printf("Buffer is not freed :)\n");
}

class ComputingServer {
public:
    ComputingServer(unsigned dPort_,const std::string& wServersFile,unsigned wPort_):
        dPort(dPort_),
        wPort(wPort_),
        dataSocket(ctx, ZMQ_REP),
        weightSocket(ctx, ZMQ_DEALER){
            loadWeightServers(weightServerAddrs,wServersFile);
            printf("%zu\n",weightServerAddrs.size() );
            printf("LOADING WSERVER FILE\n");
            for(int i=0;i<weightServerAddrs.size();++i){
                printf("%s\n",weightServerAddrs[i] );
            }

    }

    //read weight file 
    void loadWeightServers(std::vector<char *>& addresses, const std::string& wServersFile);

    // Keep listening to computing requests
    void run();

    // Sending and Requesting functions
    Matrix requestWeightsMatrix(zmq::socket_t& socket, unsigned layer);
    void sendMatrices(Matrix& zResult, Matrix& actResult, zmq::socket_t& socket, int32_t id);


private:
    //ntw related objs
    zmq::context_t ctx;
    zmq::socket_t dataSocket;
    zmq::socket_t weightSocket;

    std::vector<char*> weightServerAddrs;
    unsigned dPort;
    unsigned wPort;
    
    ComputingUnit cu;
};


void ComputingServer::run(){

    //use port as ipc addresss
    char ipc_addr[50];
    sprintf(ipc_addr, "ipc:///tmp/GPU_COMM:%u", dPort); 
    dataSocket.bind("tcp://*:1234");


    std::cout << "Binding computing server to " << ipc_addr << "..." << std::endl;
    dataSocket.bind(ipc_addr);

    // Keeps listening on coord's requests.
    std::cout << "[GPU] Starts listening for GPU requests from DATASERVER..." << std::endl;

    try {
        bool terminate = false;
        while (!terminate) {
            zmq::message_t header;
            zmq::message_t confirm(5);
            zmq::message_t aggreChunk;
            unsigned op;
            unsigned layer;
            unsigned ROWS,COLS;

            dataSocket.recv(&header);
            op = parse<unsigned>((char *) header.data(), 0);
            layer = parse<unsigned>((char *) header.data(), 1);
            ROWS = parse<unsigned>((char *) header.data(), 2);
            COLS = parse<unsigned>((char *) header.data(), 3);

            printf("op %u\n", op);
            printf("Layer %u\n", layer);
            printf("ROWS %u\n", ROWS);
            printf("COLS %u\n", COLS);
            dataSocket.send(confirm);


            std::vector<Matrix> weights;
            std::vector<std::thread> wThreads;
            for(size_t i=0;i<weightServerAddrs.size();i++){
                wThreads.push_back(
                    std::thread([&] (size_t index){
                        printf("for %zu\n",index);
                        unsigned id = 0; // id=0 for GPU
                        unsigned ipc_addr_len=strlen(ipc_addr);
                        size_t identity_len = sizeof(unsigned) + ipc_addr_len;
                        char identity[identity_len];
                        memcpy(identity, (char *) &id, sizeof(unsigned));
                        memcpy(identity + sizeof(unsigned), ipc_addr, ipc_addr_len);

                        std::cout << "< GPU SERVER FORWARD > Asking weightserver..." << std::endl;
                        weightSocket.setsockopt(ZMQ_IDENTITY, identity, identity_len);
                        char whost_port[50];
                        sprintf(whost_port, "tcp://%s:%u", weightServerAddrs.at(index), wPort);
                        printf("connect to %s\n", whost_port);
                        weightSocket.connect(whost_port);
                        weights.push_back(requestWeightsMatrix(weightSocket, layer));
                        std::cout << "<  GPU SERVER FORWARD > Got data from weightserver." << std::endl;
                },i)
            );}

            std::cout << "< GPU SERVER FORWARD  > Getting data from dataserver..." << std::endl;
            dataSocket.recv(&aggreChunk);
            printf("Chunk Received\n");
            printf("%zu\n", aggreChunk.size());
            // printf("%u\n", ROWS*COLS);
            dataSocket.send(confirm);
            
            // zmq::socket_t dSocket(ctx, ZMQ_DEALER);
            // dataSocket.setsockopt(ZMQ_IDENTITY, identity, identity_len);
            // char dhost_port[50];
            // sprintf(dhost_port, "tcp://%s:%s", dataserver, dport);
            // dSocket.connect(dhost_port);
            // Matrix feats = requestMatrix(dSocket, id);
            std::cout << "< GPU SERVER FORWARD > Got data from dataserver." << std::endl;
            for(auto &t: wThreads)
                t.join();
            // break;
            // Matrix z = cu.dot(feats, weights);
            // Matrix act = cu.activate(z);
            // sendMatrices(z,act,dSocket,id);
        }
    } catch (std::exception& ex) {
        std::cerr << "[ERROR] " << ex.what() << std::endl;
    }
}


//Send multiplied matrix result back to dataserver.
void ComputingServer::sendMatrices(Matrix& zResult, Matrix& actResult, zmq::socket_t& socket, int32_t id) {
        // // Send push header.
        // zmq::message_t header(HEADER_SIZE);
        // populateHeader((char *) header.data(), OP::PUSH, id, zResult.rows, zResult.cols);
        // socket.send(header, ZMQ_SNDMORE);

        // // Send zData and actData.
        // zmq::message_t zData(zResult.getDataSize());
        // std::memcpy(zData.data(), zResult.getData(), zResult.getDataSize());
        // zmq::message_t actData(actResult.getDataSize());
        // std::memcpy(actData.data(), actResult.getData(), actResult.getDataSize());
        // socket.send(zData, ZMQ_SNDMORE);
        // socket.send(actData);

        // // Wait for data settled reply.
        // zmq::message_t confirm;
        // socket.recv(&confirm);
}



void ComputingServer::loadWeightServers(std::vector<char *>& addresses, const std::string& wServersFile){
    std::ifstream infile(wServersFile);
    if (!infile.good())
        printf("Cannot open weight server file: %s [Reason: %s]\n", wServersFile.c_str(), std::strerror(errno));

    assert(infile.good());

    std::string line;
    while (!infile.eof()) {
        std::getline(infile, line);
        boost::algorithm::trim(line);

        if (line.length() == 0)
            continue;   
        
        char *addr = strdup(line.c_str());
        addresses.push_back(addr);
    }
}




/**
 *
 * Request the input matrix data from weightserver.
 * 
 */
//TODO: Can be modify to zerocopy
Matrix
ComputingServer::requestWeightsMatrix(zmq::socket_t& socket, unsigned layer) {
    
    // Send pull request.
    zmq::message_t header(HEADER_SIZE);
    populateHeader((char *) header.data(), OP::PULL_FORWARD, layer);
    socket.send(header);

    // Listen on respond.
    zmq::message_t respHeader;
    socket.recv(&respHeader);

    // Parse the respond.
    unsigned layerResp = parse<unsigned>((char *) respHeader.data(), 1);
    if ((int)layerResp == -1) {      // Failed.
        std::cerr << "[ ERROR ] No corresponding matrix chunk!" << std::endl;
        return Matrix();
    } else {                    // Get matrices data.
        unsigned rows = parse<unsigned>((char *) respHeader.data(), 2);
        unsigned cols = parse<unsigned>((char *) respHeader.data(), 3);
        zmq::message_t matxData(rows * cols * sizeof(float));
        socket.recv(&matxData);

        char *matxBuffer = new char[matxData.size()];
        std::memcpy(matxBuffer, matxData.data(), matxData.size());

        Matrix m(rows, cols, matxBuffer);
        return m;
    }
}

#endif 
