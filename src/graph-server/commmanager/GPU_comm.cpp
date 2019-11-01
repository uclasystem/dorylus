#include "GPU_comm.hpp"

static void doNotFreeBuffer(void *data, void *hint){
    // printf("Buffer is not freed :)\n");
}

extern "C" ResourceComm* createComm(CommInfo& commInfo){
    return new GPUComm(commInfo.nodeId,commInfo.numNodes,commInfo.dataserverPort,
                        commInfo.wServersFile,commInfo.weightserverPort);
}

GPUComm::GPUComm(unsigned nodeId_, unsigned numNodes_, unsigned dataserverPort_,const std::string& wServersFile_,unsigned wPort_):
        ResourceComm(),
        confirm(5),
        wServersFile(wServersFile_),
        nodeId(nodeId_),
        numNodes(numNodes_),
        dPort(dataserverPort_),
        wPort(wPort_),
        dataSocket(ctx,ZMQ_REQ){

        eval=0;

        ComputingServer* comp_server=new ComputingServer(this);
        printLog(nodeId,"Starting thread\n");
        comp_server_thread=std::thread(std::bind(&ComputingServer::run,comp_server));
        
        char ipc_addr[50];
        sprintf(ipc_addr, "inproc://%u", dPort);
        
        dataSocket.connect(ipc_addr);
        printLog(nodeId,"Connecting to %s\n", ipc_addr);
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

void GPUComm::requestForward(unsigned layer,bool lastLayer){

    try {
        // float split_data=split;
        // if(!eval)
        //     split_data=0;
        // else
        //     printLog(nodeId,"Evaling");
    
        int op=OP::REQ_FORWARD;
        unsigned lastLayerInt=lastLayer;
        sendFourBytes((char*)&op);
        sendFourBytes((char*)&layer);
        sendFourBytes((char*)&lastLayerInt);
        // sendFourBytes((char*)&split_data);
        sendMatrix(actMatrix);
        sendResultPtr(zData);
        sendResultPtr(actData);
        unsigned done=0;
        sendFourBytes((char*)&done);


        // if(eval){
        //     unsigned numValidationVertices=std::ceil(split*numLocalVertices);
        //     sendMatrix(targetMatrix);
        //     unsigned totalCorrect = requestFourBytes<unsigned>();
        //     float loss = requestFourBytes<float>();
        //     printLog(nodeId, "Accuracy this epoch: %f",(float) totalCorrect / (float) numValidationVertices);
        //     printLog(nodeId, "Loss this epoch %f",loss / (float) numValidationVertices);
        // }
    }
    catch(std::exception& ex){
        std::cerr << "[ERROR] " << ex.what() << std::endl;
    }
}


void GPUComm::setTrainValidationSplit(float trainPortion, unsigned numLocalVertices){
    split=trainPortion;
};


// For backward-prop.
void GPUComm::newContextBackward(FeatType **zBufs, FeatType **actBufs, FeatType *targetBuf,
                            unsigned numLocalVertices, std::vector<unsigned> layerConfig){
    printLog(nodeId, "***********SHOULD NOT APPEAR********************");
    printLog(nodeId, "GPU BACKWARD context(Old) created.");
}

void GPUComm::newContextBackward(FeatType *oldGradBuf, FeatType *newGradBuf, std::vector<Matrix> *savedTensors, FeatType *targetBuf,
                                    unsigned numLocalVertices, unsigned inFeatDim, unsigned outFeatDim, unsigned targetDim){
    // Create new matrices object for workers to access.
    oldGradMatrix=Matrix(numLocalVertices, outFeatDim, oldGradBuf);
    newGradMatrix=Matrix(numLocalVertices, inFeatDim, newGradBuf);
    targetMatrix=Matrix(numLocalVertices, targetDim, targetBuf);
    this->savedTensors=savedTensors;
    printLog(nodeId, "GPU BACKWARD context created.");
}

void GPUComm::requestBackward(unsigned numLayers, bool lastLayer){
    printLog(nodeId, "GPU BACKWARD request.");
    unsigned layer=numLayers;
    try {
        int lastLayerInt=(int)lastLayer;
        int op=OP::REQ_BACKWARD;
        sendFourBytes((char*)&op);
        sendFourBytes((char*)&numLayers);//which layer |not the totalnumber of layers
        sendFourBytes((char*)&numNodes);
        sendFourBytes((char*)&lastLayerInt);
        if(lastLayer){
            sendMatrix(savedTensors[layer][TYPE::ACT - 1]);
            sendMatrix(targetMatrix);
            sendResultPtr(newGradMatrix.getData());
            sendMatrix(savedTensors[layer][TYPE::AH - 1]);
        }else{
            sendMatrix(oldGradMatrix);
            sendMatrix(savedTensors[layer][TYPE::Z - 1]);
            sendResultPtr(newGradMatrix.getData());
            sendMatrix(savedTensors[layer][TYPE::AH - 1]);
        }
        unsigned done=0;
        sendFourBytes((char*)&done);
    }
    catch(std::exception& ex){
        std::cerr << "[ERROR] " << ex.what() << std::endl;
    }
}


void GPUComm::sendShutdownMessage(){
    printLog(nodeId, "Send Shutdown Message\n");

     // Send kill message.
    zmq::message_t header(HEADER_SIZE);
    populateHeader((char *) header.data(), OP::TERM);
    dataSocket.send(header);
    dataSocket.recv(&confirm);

    comp_server_thread.join();
    printLog(nodeId, "Joined\n");
}

void GPUComm::sendFourBytes(char* data){
    zmq::message_t dataMsg(4);
    memcpy(dataMsg.data(),data,4);
    dataSocket.send(dataMsg);
    dataSocket.recv(&confirm);
}

template <class T>
T GPUComm::requestFourBytes(){
    zmq::message_t header(4);
    dataSocket.recv(&header);
    dataSocket.send(confirm);
    return *((T*)header.data());
}

void GPUComm::sendMatrix(Matrix& m){
    zmq::message_t matrixMsg(HEADER_SIZE);
    populateHeader((char *) matrixMsg.data(), m.getRows(), m.getCols());
    char* dataPtr=(char*)m.getData();
    
    std::memcpy((char*)matrixMsg.data()+sizeof(unsigned)*2, (char*) &dataPtr,
        sizeof(FeatType*));
    
    dataSocket.send(matrixMsg);
    dataSocket.recv(&confirm);
}

void GPUComm::sendResultPtr(FeatType* ptr){
    zmq::message_t ptrMsg(sizeof(FeatType*));
    memcpy(ptrMsg.data(),&ptr,sizeof(FeatType*));
    dataSocket.send(ptrMsg);
    dataSocket.recv(&confirm);
}