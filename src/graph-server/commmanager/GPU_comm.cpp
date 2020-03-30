#include "GPU_comm.hpp"

static void doNotFreeBuffer(void *data, void *hint) {
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
        currLayer(0),
        dPort(dataserverPort_),
        wPort(wPort_) {
            comp_server=new ComputingServer(this);
}

// For forward.
void GPUComm::newContext(unsigned layer, Matrix &inputTensor_, Matrix &outputTensor_, std::vector<Matrix> *savedTensors_) {
    currLayer = layer;

    inputTensor = inputTensor_;
    outputTensor = outputTensor_;
    savedTensors = savedTensors_;

    // printLog(nodeId, "GPU FORWARD context created.");
}
// For backward-prop.
void GPUComm::newContext(unsigned layer, Matrix &inputTensor_, Matrix &outputTensor_, Matrix &targetTensor_, std::vector<Matrix> *savedTensors_) {
    // Create a new matrix object for workers to access.
    currLayer = layer;

    inputTensor = inputTensor_;
    outputTensor = outputTensor_;
    targetTensor = targetTensor_;
    savedTensors = savedTensors_;
}

void GPUComm::requestForward(unsigned layer, bool lastLayer) {
    comp_server->processForward(layer,lastLayer);
}

void GPUComm::requestBackward(unsigned layer, bool lastLayer) {
    printLog(nodeId, "GPU BACKWARD request. %u",layer);
    comp_server->processBackward(layer, lastLayer);
}


void GPUComm::sendShutdownMessage() {
    printLog(nodeId, "Send Shutdown Message\n");
    // Send kill message.
    comp_server->terminate();
}
