#ifndef __COMP_SERVER_HPP__
#define __COMP_SERVER_HPP__

#include <zmq.hpp>
#include <chrono>
#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <sstream>
#include <boost/algorithm/string/trim.hpp>
#include <vector>
#include <thread>
#include "../comp_unit/comp_unit.hpp"
#include "../../../src/utils/utils.hpp"
#include "../../../src/graph-server/utils/utils.cpp"


const float LEARNING_RATE=0.1;

/** Struct for wrapping over the returned matrices. */
typedef struct {
    std::vector<Matrix> zMatrices;          // Layer 1 -> last.
    std::vector<Matrix> actMatrices;        // Layer 0 -> last.
    Matrix targetMatrix;
} GraphData;


class ComputingServer {
public:
    ComputingServer(unsigned dPort_,const std::string& wServersFile,unsigned wPort_);

    //read weight file 
    void loadWeightServers(std::vector<char *>& addresses, const std::string& wServersFile);

    // Keep listening to computing requests
    void run();

    // Sending and Requesting functions
    Matrix requestWeightsMatrix( unsigned layer);
    Matrix requestFeatsMatrix(unsigned rows,unsigned cols);
    void sendMatrices(Matrix& zResult, Matrix& actResult);
    void processForward(zmq::message_t &header);

    //for backward
    GraphData requestForwardMatrices(unsigned numLayers);
    std::vector<Matrix> requestWeightsMatrices(unsigned numLayers);
    void processBackward(zmq::message_t &header);
    void sendInfoMessage(zmq::socket_t& weightsocket, unsigned numLambdas);

private:
    //ntw related objs
    zmq::context_t dctx;
    zmq::context_t wctx;
    zmq::socket_t dataSocket;
    zmq::socket_t weightSocket;

    std::vector<char*> weightServerAddrs;
    unsigned dPort;
    unsigned wPort;
    unsigned nodeId;

    char ipc_addr[50];

    //GPU part
    ComputingUnit cu;
};

void doNotFreeBuffer(void *data, void *hint){
    // printf("Buffer is not freed :)\n");
}

ComputingServer::ComputingServer(unsigned dPort_,const std::string& wServersFile,unsigned wPort_):
    dPort(dPort_),
    wPort(wPort_),
    dataSocket(dctx, ZMQ_REP),
    weightSocket(wctx, ZMQ_DEALER){
        loadWeightServers(weightServerAddrs,wServersFile);

        //use port as ipc addresss
        sprintf(ipc_addr, "ipc:///tmp/GPU_COMM:%u", dPort); 
        std::cout << "Binding computing server to " << ipc_addr << "..." << std::endl;
        dataSocket.bind(ipc_addr);

}

void ComputingServer::run(){
    
    // Keeps listening on coord's requests.
    std::cout << "[GPU] Starts listening for GPU requests from DATASERVER..." << std::endl;

    zmq::message_t confirm(5);
    zmq::message_t init_header(HEADER_SIZE);
    dataSocket.recv(&init_header);
    nodeId = parse<unsigned>((char *) init_header.data(), 0);
    dataSocket.send(confirm);

    try {
        bool terminate=0;
        while (!terminate) {
            printf("next op\n");
            zmq::message_t header(HEADER_SIZE);
            dataSocket.recv(&header);
            unsigned op = parse<unsigned>((char *) header.data(), 0);
            dataSocket.send(confirm);

            switch (op){
                case OP::TERM:
                printf("Terminating\n");
                terminate=1;
                break;
                case OP::REQ_FORWARD:
                processForward(header);
                break;
                case OP::REQ_BACKWARD:
                processBackward(header);
                break;
                default:
                printf("unknown OP\n");
            }
        }
    } catch (std::exception& ex) {
        std::cerr << "[ERROR] " << ex.what() << std::endl;
    }
}


void ComputingServer::sendInfoMessage(zmq::socket_t& weightsocket, unsigned numLambdas) {
    zmq::message_t header(HEADER_SIZE);
    populateHeader((char *) header.data(), OP::INFO, numLambdas);
    weightsocket.send(header);

    // Wait for info received reply.
    zmq::message_t confirm;
    weightsocket.recv(&confirm);
}


void ComputingServer::processBackward(zmq::message_t &header){
    
    zmq::message_t confirm(5);
    unsigned layer= parse<unsigned>((char *) header.data(), 1);
    unsigned numNode = parse<unsigned>((char *) header.data(), 2);

    //send INFO to weight server
    printf("Send to weight \n");
    if(nodeId<weightServerAddrs.size()){
        unsigned count = 0; 
        for (size_t i=0;i<numNode;++i)
            if(i%weightServerAddrs.size()==nodeId)
                count+=1;
        sendInfoMessage(weightSocket, count);
    }
        
    requestForwardMatrices(layer);

}

/**
 *
 * Request the graph feature matrices data from dataserver.
 * 
 */ 
GraphData
ComputingServer::requestForwardMatrices(unsigned numLayers) {
    zmq::message_t confirm(5);    
    // Send pull request.
    zmq::message_t header(HEADER_SIZE);
    populateHeader((char *) header.data(), OP::PULL_BACKWARD, 0);
    // printf("Send to h \n");
    // dataSocket.send(header);
    // printf("1\n");
    GraphData graphData;

    // Receive z matrices chunks, from layer 1 -> last.
    for (size_t i = 1; i <= numLayers; ++i) {
        zmq::message_t respHeader;
        dataSocket.recv(&respHeader);

        unsigned rows = parse<unsigned>((char *) respHeader.data(), 2);
        unsigned cols = parse<unsigned>((char *) respHeader.data(), 3);
        zmq::message_t matxData(rows * cols * sizeof(FeatType));
        dataSocket.recv(&matxData);

        FeatType *matxBuffer = new FeatType[rows * cols];
        std::memcpy(matxBuffer, matxData.data(), matxData.size());

        graphData.zMatrices.push_back(Matrix(rows, cols, matxBuffer));
    }

    // Receive act matrices chunks, from layer 0 -> last.
    for (size_t i = 0; i <= numLayers; ++i) {
        zmq::message_t respHeader;
        dataSocket.recv(&respHeader);
        
        unsigned layerResp = parse<unsigned>((char *) respHeader.data(), 1);
       
       
        unsigned rows = parse<unsigned>((char *) respHeader.data(), 2);
        unsigned cols = parse<unsigned>((char *) respHeader.data(), 3);
        zmq::message_t matxData(rows * cols * sizeof(FeatType));
        dataSocket.recv(&matxData);

        FeatType *matxBuffer = new FeatType[rows * cols];
        std::memcpy(matxBuffer, matxData.data(), matxData.size());

        graphData.actMatrices.push_back(Matrix(rows, cols, matxBuffer));
    
    }

    // Receive target label matrix chunk.
    zmq::message_t respHeader;
    dataSocket.recv(&respHeader);

    unsigned layerResp = parse<unsigned>((char *) respHeader.data(), 1);
    
    unsigned rows = parse<unsigned>((char *) respHeader.data(), 2);
    unsigned cols = parse<unsigned>((char *) respHeader.data(), 3);
    zmq::message_t matxData(rows * cols * sizeof(FeatType));

    dataSocket.recv(&matxData);
    dataSocket.send(confirm);

    FeatType *matxBuffer = new FeatType[rows * cols];
    std::memcpy(matxBuffer, matxData.data(), matxData.size());

    graphData.targetMatrix = Matrix(rows, cols, matxBuffer);

    return graphData;
}



void ComputingServer::processForward(zmq::message_t &header){
    unsigned layer = parse<unsigned>((char *) header.data(), 1);
    unsigned rows = parse<unsigned>((char *) header.data(), 2);
    unsigned cols = parse<unsigned>((char *) header.data(), 3);

    Matrix weights;
    std::thread wThread=std::thread([&]{
        unsigned ipc_addr_len=strlen(ipc_addr);
        size_t identity_len = sizeof(unsigned) + ipc_addr_len;
        char identity[identity_len];
        memcpy(identity, (char *) &nodeId, sizeof(unsigned));
        memcpy(identity + sizeof(unsigned), ipc_addr, ipc_addr_len);

        std::cout << "< GPU SERVER FORWARD > Asking weightserver..." << std::endl;
        weightSocket.setsockopt(ZMQ_IDENTITY, identity, identity_len);
        char whost_port[50];
        sprintf(whost_port, "tcp://%s:%u", weightServerAddrs.at(nodeId%weightServerAddrs.size()), wPort);
        printf("connect to %s\n", whost_port);
        weightSocket.connect(whost_port);

        weights=requestWeightsMatrix(layer);
        std::cout << "<  GPU SERVER FORWARD > Got data from weightserver." << std::endl;
    });
    
    Matrix feats=requestFeatsMatrix(rows,cols);
    wThread.join();
    printf("Finish receiving all data\n");
    printf("Start Feat .* Weight\n");
    double t1=getTimer();
    CuMatrix z = cu.dot(feats, weights);
    printf("Start Act(Z)\n");
    FeatType* z_buffer=new FeatType[z.getRows()*z.getCols()];
    memcpy(z_buffer,z.getData(),z.getDataSize());
    Matrix z_send(z.getRows(),z.getCols(),z_buffer);            
    cu.activate(z);//z data get activated ...
    printf("Total*: %lf\n", getTimer()-t1);
    printf("Sending Z and Act Z\n");
    sendMatrices(z_send,z);

    delete[] (feats.getData());
    delete[] (z_buffer);
}

//Send multiplied matrix result back to dataserver.
void ComputingServer::sendMatrices(Matrix& zResult, Matrix& actResult) {
        zmq::message_t confirm;
        // // Send push header.
        zmq::message_t header(HEADER_SIZE);
        populateHeader((char *) header.data(),zResult.getRows(), zResult.getCols());
        dataSocket.send(header);
        dataSocket.recv(&confirm);
        // // Send zData and actData.
        zmq::message_t zData(zResult.getData(),zResult.getDataSize(),doNotFreeBuffer, NULL);
        dataSocket.send(zData);
        dataSocket.recv(&confirm);
        zmq::message_t actData(actResult.getData(),actResult.getDataSize(),doNotFreeBuffer, NULL);
        dataSocket.send(actData);
        dataSocket.recv(&confirm);
        dataSocket.send(confirm); 
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


Matrix
ComputingServer::requestFeatsMatrix(unsigned rows,unsigned cols) {
    zmq::message_t confirm(5);
    zmq::message_t aggreChunk;
    std::cout << "< GPU SERVER FORWARD  > Getting data from dataserver..." << std::endl;
    dataSocket.recv(&aggreChunk);
    std::cout << "< GPU SERVER FORWARD > Got data from dataserver." << std::endl;
    FeatType * feats_buffer=new FeatType[rows * cols];
    memcpy((char*)feats_buffer,(char*)aggreChunk.data(),rows * cols * sizeof(FeatType));
    Matrix m(rows, cols,feats_buffer);
    return m;
}

/**
 *
 * Request the input matrix data from weightserver.
 * 
 */
Matrix
ComputingServer::requestWeightsMatrix( unsigned layer) {
    
    // Send pull request.
    zmq::message_t header(HEADER_SIZE);
    populateHeader((char *) header.data(), OP::PULL_FORWARD, layer);
    weightSocket.send(header);

    // Listen on respond.
    zmq::message_t respHeader(HEADER_SIZE);
    weightSocket.recv(&respHeader);

    // Parse the respond.
    unsigned layerResp = parse<unsigned>((char *) respHeader.data(), 1);
    if ((int)layerResp == -1) {      // Failed.
        std::cerr << "[ ERROR ] No corresponding matrix chunk!" << std::endl;
        return Matrix();
    } else {                    // Get matrices data.
        unsigned rows = parse<unsigned>((char *) respHeader.data(), 2);
        unsigned cols = parse<unsigned>((char *) respHeader.data(), 3);
        zmq::message_t wData(rows * cols * sizeof(FeatType));
        weightSocket.recv(&wData);
        FeatType *wBuffer = new FeatType[rows*cols];
        memcpy((char*)wBuffer,(char*)wData.data(),rows * cols * sizeof(FeatType));
        Matrix m(rows, cols, wBuffer);
        return m;
    }
}






#endif 
