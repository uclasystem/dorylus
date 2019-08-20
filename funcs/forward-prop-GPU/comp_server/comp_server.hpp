#ifndef __COMP_SERVER_HPP__
#define __COMP_SERVER_HPP__

#include <zmq.hpp>
#include <chrono>
#include <iostream>
#include <fstream>
// #include "../comp_unit/comp_unit.hpp"
#include <string>
#include <cstring>
#include <sstream>
#include "../../../src/utils/utils.hpp"

class ComputingServer {
public:
    ComputingServer(unsigned dPort_,std::string weightServerIp_, unsigned wPort_):
        dPort(dPort_),
        weightServerIp(weightServerIp_),
        wPort(wPort_),
        dataSocket(ctx, ZMQ_REP),
        weightSocket(ctx, ZMQ_DEALER){

    }

    // Keep listening to computing requests
    void run();

    // Sending and Requesting functions
    Matrix requestMatrix(zmq::socket_t& socket, int32_t id);
    void sendMatrices(Matrix& zResult, Matrix& actResult, zmq::socket_t& socket, int32_t id);


private:
    //ntw related objs
    zmq::context_t ctx;
    zmq::socket_t dataSocket;
    zmq::socket_t weightSocket;
    
    std::string weightServerIp;
    unsigned dPort;
    unsigned wPort;
    
    // ComputingUnit cu;
};


void ComputingServer::run(){

    //use port as ipc addresss
    char ipc_addr[50];
    sprintf(ipc_addr, "ipc:///tmp/GPU_COMM:%u", dPort); 
    // dataSocket.bind("tcp://*:1234");


    std::cout << "Binding computing server to " << ipc_addr << "..." << std::endl;
    dataSocket.bind(ipc_addr);

    // Keeps listening on coord's requests.
    std::cout << "[GPU] Starts listening for GPU requests from DATASERVER..." << std::endl;

    try {
        bool terminate = false;
        while (!terminate) {
            zmq::message_t w_ip_msg;
            zmq::message_t header;
            zmq::message_t header2;
            zmq::message_t confirm(5);
            zmq::message_t aggreChunk;

            dataSocket.recv(&w_ip_msg);
            dataSocket.send(confirm);


            dataSocket.recv(&header);
            unsigned op = parse<unsigned>((char *) header.data(), 0);
            unsigned layer = parse<unsigned>((char *) header.data(), 1);
            wPort = parse<unsigned>((char *) header.data(), 2);
            dataSocket.send(confirm);

            std::string weightServerIp(w_ip_msg.size()+1,' ');
            memcpy(&weightServerIp[0], (char *) w_ip_msg.data(), w_ip_msg.size());
            weightServerIp[w_ip_msg.size()] = '\0';

            printf("op %u\n", op);
            printf("layer %u\n", layer);
            printf("wPort %u\n", wPort);
            printf("IP %s\n", weightServerIp.c_str());

            dataSocket.recv(&header2);
            op = parse<unsigned>((char *) header2.data(), 0);
            unsigned ROWS=parse<unsigned>((char *) header2.data(), 2);
            unsigned COLS=parse<unsigned>((char *) header2.data(), 3);;
            printf("op %u\n", op);
            printf("ROWS %u\n", ROWS);
            printf("COLS %u\n", COLS);
            dataSocket.send(confirm);


            dataSocket.recv(&aggreChunk);
            printf("Chunk Received\n");
            printf("%zu\n", aggreChunk.size());
            printf("%u\n", ROWS*COLS);
            for (unsigned i=0;i<ROWS*COLS;++i){
                printf("%f\n", ((float*)aggreChunk.data())[i]);    
            }
            dataSocket.send(confirm);

            // Matrix weights;
            // std::thread t([&] {
            //     std::cout << "< matmul > Asking weightserver..." << std::endl;
            //     zmq::socket_t wSocket(ctx, ZMQ_DEALER);
            //     wSocket.setsockopt(ZMQ_IDENTITY, identity, identity_len);
            //     char whost_port[50];
            //     sprintf(whost_port, "tcp://%s:%s", weightserver, wport);
            //     wSocket.connect(whost_port);
            //     weights = requestMatrix(wSocket, layer);
            //     std::cout << "< matmul > Got data from weightserver." << std::endl;
            // });

            // std::cout << "< matmul > Asking dataserver..." << std::endl;
            // zmq::socket_t dSocket(ctx, ZMQ_DEALER);
            // dataSocket.setsockopt(ZMQ_IDENTITY, identity, identity_len);
            // char dhost_port[50];
            // sprintf(dhost_port, "tcp://%s:%s", dataserver, dport);
            // dSocket.connect(dhost_port);
            // Matrix feats = requestMatrix(dSocket, id);
            // std::cout << "< matmul > Got data from dataserver." << std::endl;
            // t.join();

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

// Request the input matrix data from dataserver.
Matrix ComputingServer::requestMatrix(zmq::socket_t& socket, int32_t id) {
    return Matrix();

    // // Send pull request.
    // zmq::message_t header(HEADER_SIZE);
    // populateHeader((char *) header.data(), OP::PULL, id);
    // socket.send(header);

    // // Listen on respond.
    // zmq::message_t respHeader;
    // socket.recv(&respHeader);

    // // Parse the respond.
    // int32_t layerResp = parse<int32_t>((char *) respHeader.data(), 1);
    // if (layerResp == -1) {      // Failed.
    //     std::cerr << "[ ERROR ] No corresponding matrix chunk!" << std::endl;
    //     return Matrix();
    // } else {                    // Get matrix data.
    //     int32_t rows = parse<int32_t>((char *) respHeader.data(), 2);
    //     int32_t cols = parse<int32_t>((char *) respHeader.data(), 3);
    //     zmq::message_t matxData(rows * cols * sizeof(FeatType));
    //     socket.recv(&matxData);

    //     char *matxBuffer = new char[matxData.size()];
    //     std::memcpy(matxBuffer, matxData.data(), matxData.size());

    //     Matrix m(rows, cols, matxBuffer);
    //     return m;
    // }
}


#endif 
