#include "GPU_comm.hpp"
#include <unistd.h>

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
        ready=0;
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

void GPUComm::requestForward(unsigned layer){

    try {
        float split_data=split;
        if(!eval)
            split_data=0;
        else
            printLog(nodeId,"Evaling");

        int op=OP::REQ_FORWARD;
        sendFourBytes((char*)&op);
        sendFourBytes((char*)&layer);
        sendFourBytes((char*)&split_data);
        sendMatrix(actMatrix);
        sendResultPtr(zData);
        sendResultPtr(actData);
        
        if(eval){
            unsigned numValidationVertices=std::ceil(split*numLocalVertices);
            sendMatrix(targetMatrix);
            unsigned totalCorrect = requestFourBytes<unsigned>();
            float loss = requestFourBytes<float>();
            printLog(nodeId, "Accuracy this epoch: %f",(float) totalCorrect / (float) numValidationVertices);
            printLog(nodeId, "Loss this epoch %f",loss / (float) numValidationVertices);
        }
        unsigned done=0;
        sendFourBytes((char*)&done);
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
    zMatrices.clear();
    for (size_t i = 1; i < layerConfig.size(); ++i)
        zMatrices.push_back(Matrix(numLocalVertices, layerConfig[i], zBufs[i]));
    actMatrices.clear();
    for (size_t i = 0; i < layerConfig.size(); ++i)
        actMatrices.push_back(Matrix(numLocalVertices, layerConfig[i], actBufs[i]));
    targetMatrix=Matrix(numLocalVertices, layerConfig[layerConfig.size() - 1], targetBuf);

    printLog(nodeId, "GPU BACKWARD context created.");
}


void GPUComm::requestBackward(unsigned numLayers, bool lastLayer){
    printLog(nodeId, "GPU BACKWARD request.");

    try {
        int op=OP::REQ_BACKWARD;
        sendFourBytes((char*)&op);
        sendFourBytes((char*)&numLayers);
        sendFourBytes((char*)&numNodes);
        sendBackpropChunks();
    }
    catch(std::exception& ex){
        std::cerr << "[ERROR] " << ex.what() << std::endl;
    }
}


void GPUComm::sendBackpropChunks(){

    
    // Send z matrices, from layer 1-> last.
    for (Matrix& matrix : zMatrices)
        sendMatrix(matrix);

    // Send activation matrices, from layer 0 -> last.
    for (Matrix& matrix : actMatrices) 
        sendMatrix(matrix);

    // Send target label matrix.
    sendMatrix(targetMatrix);
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