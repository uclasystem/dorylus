#include "GPU_comm.hpp"

void doNotFreeBuffer(void *data, void *hint){
    printf("Buffer is not freed :)\n");
}

void GPUComm::newContextForward(FeatType *dataBuf, FeatType *zData_, FeatType *actData_,
                              unsigned numLocalVertices, unsigned numFeats, unsigned numFeatsNext_){
    
    // Create a new matrix object for workers to access.
    actMatrix=Matrix(numLocalVertices, numFeats, dataBuf);
    zData = zData_;
    actData = actData_;
    numFeatsNext = numFeatsNext_;
    // printLog(nodeId, "GPU FORWARD context created.");

}

void GPUComm::requestForward(unsigned layer){
    try {
        zmq::message_t confirm(5);
        zmq::message_t header(HEADER_SIZE);

        unsigned actRows=actMatrix.getRows();
        unsigned actCols=actMatrix.getCols();


        populateHeader((char *) header.data(), OP::REQ_FORWARD, layer,actRows,actCols);
        dataSocket.send(header);
        dataSocket.recv(&confirm);

        zmq::message_t dataMsg(actMatrix.getData(), actRows*actCols, doNotFreeBuffer, NULL);
        dataSocket.send(dataMsg);

        zmq::message_t resultHeader(HEADER_SIZE);
        dataSocket.recv(&resultHeader);
        unsigned newActRows=parse<unsigned>((char *) header.data(), 0);
        unsigned newActCols=parse<unsigned>((char *) header.data(), 1);
        unsigned recvSize=newActRows*newActCols*sizeof(FeatType);
        zmq::message_t newZ(zData,recvSize,doNotFreeBuffer,NULL);
        dataSocket.recv(&newZ);
        zmq::message_t newAct(actData,recvSize,doNotFreeBuffer,NULL);
        dataSocket.recv(&newAct);

        // dataSocket.send(&confirm);

    }
    catch(std::exception& ex){
        std::cerr << "[ERROR] " << ex.what() << std::endl;
    }
}


void GPUComm::sendShutdownMessage(){
     // Send kill message.
    zmq::message_t confirm(5);
    zmq::message_t header(HEADER_SIZE);
    populateHeader((char *) header.data(), OP::TERM);
    dataSocket.send(header);
    dataSocket.recv(&confirm);
}