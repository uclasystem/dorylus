#include <zmq.hpp>
#include <chrono>
#include <iostream>
#include <fstream>
#include "../../src/utils/utils.hpp"
#include "comp_unit.hpp"

class ComputingServer {
public:

    ComputingServer(char *compserverPort_)
        : compserverPort(compserverPort_) {}

    // Keep listening to computing requests
    void run();

    // Sending and Requesting functions
    Matrix requestMatrix(zmq::socket_t& socket, int32_t id);
    void sendMatrices(Matrix& zResult, Matrix& actResult, zmq::socket_t& socket, int32_t id);


private:
    char *compserverPort;
    ComputingUnit cu;
};

void ComputingServer::run(){
    zmq::context_t ctx();
    zmq::socket_t frontend(ctx, ZMQ_REP);
    char host_port[50];
    sprintf(host_port, "tcp://*:%s", compserverPort);
    std::cout << "Binding computing server to " << host_port << "..." << std::endl;
    frontend.bind(host_port);

    // Keeps listening on coord's requests.
    std::cout << "[GPU] Starts listening for coords's requests..." << std::endl;

    try {
        bool terminate = false;
        while (!terminate) {

            // Wait on requests.
            zmq::message_t dataserverIp;
            zmq::message_t weightserverIp;
        
            //can be packed into header. do it later.
            zmq::message_t dPort;
            zmq::message_t wPort;
            zmq::message_t layer;
            zmq::message_t id;
            frontend.recv(&header);
            frontend.recv(&dataserverIp);
            frontend.recv(&weightserverIp);
            frontend.recv(&layer);
            frontend.recv(&id);

            requestMatrix();
            cu.compute();
            sendMatrices();
        }
    } catch (std::exception& ex) {
        std::cerr << "[ERROR] " << ex.what() << std::endl;
    }
}

//Send multiplied matrix result back to dataserver.
void ComputingServer::sendMatrices(Matrix& zResult, Matrix& actResult, zmq::socket_t& socket, int32_t id) {
        // Send push header.
        zmq::message_t header(HEADER_SIZE);
        populateHeader((char *) header.data(), OP::PUSH, id, zResult.rows, zResult.cols);
        socket.send(header, ZMQ_SNDMORE);

        // Send zData and actData.
        zmq::message_t zData(zResult.getDataSize());
        std::memcpy(zData.data(), zResult.getData(), zResult.getDataSize());
        zmq::message_t actData(actResult.getDataSize());
        std::memcpy(actData.data(), actResult.getData(), actResult.getDataSize());
        socket.send(zData, ZMQ_SNDMORE);
        socket.send(actData);

        // Wait for data settled reply.
        zmq::message_t confirm;
        socket.recv(&confirm);
}

// Request the input matrix data from dataserver.
Matrix ComputingServer::requestMatrix(zmq::socket_t& socket, int32_t id) { 
    // Send pull request.
    zmq::message_t header(HEADER_SIZE);
    populateHeader((char *) header.data(), OP::PULL, id);
    socket.send(header);

    // Listen on respond.
    zmq::message_t respHeader;
    socket.recv(&respHeader);

    // Parse the respond.
    int32_t layerResp = parse<int32_t>((char *) respHeader.data(), 1);
    if (layerResp == -1) {      // Failed.
        std::cerr << "[ ERROR ] No corresponding matrix chunk!" << std::endl;
        return Matrix();
    } else {                    // Get matrix data.
        int32_t rows = parse<int32_t>((char *) respHeader.data(), 2);
        int32_t cols = parse<int32_t>((char *) respHeader.data(), 3);
        zmq::message_t matxData(rows * cols * sizeof(FeatType));
        socket.recv(&matxData);

        char *matxBuffer = new char[matxData.size()];
        std::memcpy(matxBuffer, matxData.data(), matxData.size());

        Matrix m(rows, cols, matxBuffer);
        return m;
    }
}

