#include "GPU_comm.hpp"

static void doNotFreeBuffer(void *data, void *hint){
    // printf("Buffer is not freed :)\n");
}

extern "C" ResourceComm* createComm(CommInfo& commInfo) {
    return new GPUComm(commInfo.nodeId, commInfo.numNodes, commInfo.dataserverPort,
                        commInfo.wServersFile, commInfo.weightserverPort, commInfo.totalLayers);
}

extern "C" void destroyComm(GPUComm *gpuComm) {
    delete gpuComm;
}

GPUComm::GPUComm(unsigned nodeId_, unsigned numNodes_, unsigned dataserverPort_,const std::string& wServersFile_,unsigned wPort_, unsigned totalLayers_):
        ResourceComm(),
        totalLayers(totalLayers_),
        wServersFile(wServersFile_),
        nodeId(nodeId_),
        numNodes(numNodes_),
        dPort(dataserverPort_),
        wPort(wPort_),
        dataSocket(ctx,ZMQ_REQ){
            eval=0;
            comp_server=new ComputingServer(this);
}


void GPUComm::newContextForward(FeatType *dataBuf, FeatType *zData_, FeatType *actData_,
                            unsigned numLocalVertices_, unsigned numFeats, unsigned numFeatsNext_){

    // Create a new matrix object for workers to access.
    numLocalVertices=numLocalVertices_;
    actMatrix=Matrix(numLocalVertices_, numFeats, dataBuf);
    
    zData = zData_;
    actData = actData_;
    numFeatsNext = numFeatsNext_;
    printLog(nodeId, "GPU FORWARD context created.");
}

void GPUComm::requestForward(unsigned layer, bool lastLayer){
        comp_server->processForward(layer,lastLayer);
}


void GPUComm::setTrainValidationSplit(float trainPortion, unsigned numLocalVertices){
    split=trainPortion;
};

// For backward-prop.
void GPUComm::newContextBackward(FeatType *oldGradBuf, FeatType *newGradBuf, std::vector<Matrix> *savedTensors, FeatType *targetBuf,
                                    unsigned numLocalVertices, unsigned inFeatDim, unsigned outFeatDim, unsigned targetDim){
    // Create new matrices object for workers to access.
    oldGradMatrix=Matrix(numLocalVertices, outFeatDim, oldGradBuf);
    newGradMatrix=Matrix(numLocalVertices, inFeatDim, newGradBuf);
    targetMatrix=Matrix(numLocalVertices, targetDim, targetBuf);
    this->savedTensors=savedTensors;
    printLog(nodeId, "GPU BACKWARD context created.");
}

void GPUComm::requestBackward(unsigned layer, bool lastLayer){
    printLog(nodeId, "GPU BACKWARD request. %u",layer);
    comp_server->processBackward(layer, lastLayer);
}


void GPUComm::sendShutdownMessage(){
    printLog(nodeId, "Send Shutdown Message\n");
    // Send kill message.
    comp_server->terminate();
}