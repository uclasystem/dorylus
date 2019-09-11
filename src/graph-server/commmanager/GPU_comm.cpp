#include "GPU_comm.hpp"

void doNotFreeBuffer(void *data, void *hint){
    printf("Buffer is not freed :)\n");
}

void GPUComm::newContextForward(FeatType *dataBuf, FeatType *zData_, FeatType *actData_,
                              unsigned numLocalVertices_, unsigned numFeats, unsigned numFeatsNext_, bool eval_){
    // Create a new matrix object for workers to access.
    eval=eval_;
    numLocalVertices=numLocalVertices_;
    actMatrix=Matrix(numLocalVertices_, numFeats, dataBuf);
    zData = zData_;
    actData = actData_;
    numFeatsNext = numFeatsNext_;
    printLog(nodeId, "GPU FORWARD context created.");

}

void GPUComm::requestForward(unsigned layer){
    printf("requestForward \n");    
    try {
        zmq::message_t confirm(5);
        zmq::message_t header(HEADER_SIZE);

        unsigned actRows=actMatrix.getRows();
        unsigned actCols=actMatrix.getCols();
        float split_data=split;
        if(!eval){
            printLog(nodeId,"Not Evaling");
            split_data=0;
        }
        else{
            printLog(nodeId,"Evaling");
        }
        populateHeader((char *) header.data(), OP::REQ_FORWARD, layer,actRows,actCols);
        serialize<float>((char*)header.data(), 4, split_data);

        dataSocket.send(header);
        dataSocket.recv(&confirm);

        zmq::message_t dataMsg(actMatrix.getData(), actRows*actCols*sizeof(FeatType), doNotFreeBuffer, NULL);
        dataSocket.send(dataMsg);
        zmq::message_t resultHeader(HEADER_SIZE);
        dataSocket.recv(&resultHeader);
        dataSocket.send(confirm);
        unsigned newActRows=parse<unsigned>((char *) resultHeader.data(), 0);
        unsigned newActCols=parse<unsigned>((char *) resultHeader.data(), 1);
        std::size_t recvSize = newActRows*newActCols*sizeof(FeatType);
        zmq::message_t newZ;
        dataSocket.recv(&newZ);
        memcpy(zData,newZ.data(),recvSize);
        dataSocket.send(confirm);
        zmq::message_t newAct;
        dataSocket.recv(&newAct);
        memcpy(actData,newAct.data(),recvSize);
        dataSocket.send(confirm);
        dataSocket.recv(&confirm);


        if(eval){
            unsigned numValidationVertices=std::ceil(split*numLocalVertices);
            sendTargetMatrix();
            zmq::message_t result(HEADER_SIZE);
            dataSocket.recv(&result);
            unsigned totalCorrect = parse<unsigned>((char*)result.data(), 2);
            float loss = parse<float>((char*)result.data(), 3);
            
            printLog(nodeId, "Accuracy this epoch: %f",(float) totalCorrect / (float) numValidationVertices);
            printLog(nodeId, "Loss this epoch %f",loss / (float) numValidationVertices);
        }
    }
    catch(std::exception& ex){
        std::cerr << "[ERROR] " << ex.what() << std::endl;
    }
}


void GPUComm::setTrainValidationSplit(float trainPortion, unsigned numLocalVertices){
    split=trainPortion;
};


void GPUComm::sendTargetMatrix() {
    // Send target label matrix.
    zmq::message_t header(HEADER_SIZE);
    populateHeader((char *) header.data(), OP::RESP, 0, targetMatrix.getRows(), targetMatrix.getCols());
    dataSocket.send(header, ZMQ_SNDMORE);
    unsigned bufSize = targetMatrix.getRows() * targetMatrix.getCols() * sizeof(FeatType);
    zmq::message_t labelMsg( targetMatrix.getData(), bufSize,doNotFreeBuffer,NULL);
    dataSocket.send(labelMsg);
}


// For backward-prop.
void GPUComm::newContextBackward(FeatType **zBufs, FeatType **actBufs, FeatType *targetBuf,
                            unsigned numLocalVertices, std::vector<unsigned> layerConfig){
    zMatrices.clear();
    for (size_t i = 1; i < layerConfig.size(); ++i)
        zMatrices.push_back(Matrix(numLocalVertices, layerConfig[i], zBufs[i]));
    actMatrices.clear();
    for (size_t i = 0; i < layerConfig.size(); ++i)
        actMatrices.push_back(Matrix(numLocalVertices, layerConfig[i], actBufs[i]));
    targetMatrix=Matrix(numLocalVertices, layerConfig[layerConfig.size() - 1], targetBuf);

    printLog(nodeId, "GPU BACKWARD context created.");
}

void GPUComm::requestBackward(unsigned numLayers){
    printLog(nodeId, "GPU BACKWARD request.");

    try {
        zmq::message_t confirm;
        zmq::message_t header(HEADER_SIZE);

        populateHeader((char *) header.data(), OP::REQ_BACKWARD, numLayers, numNodes);
        dataSocket.send(header);
        dataSocket.recv(&confirm);
        sendBackpropChunks();
    }
    catch(std::exception& ex){
        std::cerr << "[ERROR] " << ex.what() << std::endl;
    }
}


void GPUComm::sendBackpropChunks(){
    zmq::message_t confirm(5);    
    
    // Send z matrices, from layer 1-> last.
    for (Matrix& matrix : zMatrices) {
        unsigned bufSize = matrix.getRows() * matrix.getCols() * sizeof(FeatType);
        
        zmq::message_t header(HEADER_SIZE);
        populateHeader((char *) header.data(), OP::RESP, 0, matrix.getRows(), matrix.getCols());
        dataSocket.send(header, ZMQ_SNDMORE);
        zmq::message_t zMsg(matrix.getData(),bufSize,doNotFreeBuffer,NULL);
        dataSocket.send(zMsg, ZMQ_SNDMORE);
    }

    // Send activation matrices, from layer 0 -> last.
    for (Matrix& matrix : actMatrices) {
        unsigned bufSize = matrix.getRows() * matrix.getCols() * sizeof(FeatType);

        zmq::message_t header(HEADER_SIZE);
        populateHeader((char *) header.data(), OP::RESP, 0, matrix.getRows(), matrix.getCols());
        dataSocket.send(header, ZMQ_SNDMORE);

        zmq::message_t actMsg(matrix.getData(), bufSize, doNotFreeBuffer, NULL);
        dataSocket.send(actMsg, ZMQ_SNDMORE);
    }

    // Send target label matrix.
    unsigned bufSize = targetMatrix.getRows() * targetMatrix.getCols() * sizeof(FeatType);
    zmq::message_t header(HEADER_SIZE);

    populateHeader((char *) header.data(), OP::RESP, 0, targetMatrix.getRows(), targetMatrix.getCols());
    dataSocket.send(header, ZMQ_SNDMORE);

    zmq::message_t labelMsg( targetMatrix.getData(), bufSize,doNotFreeBuffer,NULL);
    dataSocket.send(labelMsg);
    dataSocket.recv(&confirm);
}


void GPUComm::sendShutdownMessage(){
    printLog(nodeId, "Send Shutdown Message\n");

     // Send kill message.
    zmq::message_t confirm(5);
    zmq::message_t header(HEADER_SIZE);
    populateHeader((char *) header.data(), OP::TERM);
    dataSocket.send(header);
    dataSocket.recv(&confirm);
}