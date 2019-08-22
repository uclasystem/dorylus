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
        //block until computation finish
        dataSocket.recv(&confirm);

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