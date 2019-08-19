#include "comp_server.hpp"


void ComputingServer::run(){
    // zmq::context_t ctx;
    // zmq::socket_t frontend;
    
    // char host_port[50];
    // sprintf(host_port, "ipc:///tmp/feeds/%u", dataserverPort); //use port as ipc addresss

    // std::cout << "Binding computing server to " << host_port << "..." << std::endl;
    // frontend.bind(host_port);

    // // Keeps listening on coord's requests.
    // std::cout << "[GPU] Starts listening for coords's requests..." << std::endl;

    // try {
    //     bool terminate = false;
    //     while (!terminate) {

    //         // Wait on requests.
    //         zmq::message_t weightserverIp;
        
    //         //can be packed into header. do it later.
    //         zmq::message_t dPort;
    //         zmq::message_t wPort;
    //         zmq::message_t layer;
    //         zmq::message_t id;
    //         frontend.recv(&header);
    //         frontend.recv(&dataserverIp);
    //         frontend.recv(&weightserverIp);
    //         frontend.recv(&layer);
    //         frontend.recv(&id);


    //         //should make this a thread in future
    //         Matrix weights;
    //         std::thread t([&] {
    //             std::cout << "< matmul > Asking weightserver..." << std::endl;
    //             zmq::socket_t wSocket(ctx, ZMQ_DEALER);
    //             wSocket.setsockopt(ZMQ_IDENTITY, identity, identity_len);
    //             char whost_port[50];
    //             sprintf(whost_port, "tcp://%s:%s", weightserver, wport);
    //             wSocket.connect(whost_port);
    //             weights = requestMatrix(wSocket, layer);
    //             std::cout << "< matmul > Got data from weightserver." << std::endl;
    //         });

    //         std::cout << "< matmul > Asking dataserver..." << std::endl;
    //         zmq::socket_t dSocket(ctx, ZMQ_DEALER);
    //         dSocket.setsockopt(ZMQ_IDENTITY, identity, identity_len);
    //         char dhost_port[50];
    //         sprintf(dhost_port, "tcp://%s:%s", dataserver, dport);
    //         dSocket.connect(dhost_port);
    //         Matrix feats = requestMatrix(dSocket, id);
    //         std::cout << "< matmul > Got data from dataserver." << std::endl;
    //         t.join();

    //         Matrix z = cu.dot(feats, weights);
    //         Matrix act = cu.activate(z);
    //         sendMatrices(z,act,dSocket,id);
    //     }
    // } catch (std::exception& ex) {
    //     std::cerr << "[ERROR] " << ex.what() << std::endl;
    // }
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