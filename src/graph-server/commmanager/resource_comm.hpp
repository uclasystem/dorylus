#ifndef __RESOURCE_COMM_HPP__
#define __RESOURCE_COMM_HPP__
#include <string>
#include "../utils/utils.hpp"
#include "../../common/matrix.hpp"
#include <dlfcn.h>

/** For Mode*/
enum{
    LAMBDA, GPU, CPU
};

struct CommInfo {
    std::string nodeIp;
    unsigned nodeId;
    unsigned dataserverPort;
    unsigned numLambdasForward;
    unsigned numLambdasBackward;

    unsigned numNodes;
    std::string wServersFile;
    unsigned weightserverPort;

    unsigned totalLayers; //for weights prefetching

    PairQueue* queuePtr;
    TensorMap* savedVtxTensors;
    std::vector< TensorMap > *savedNNTensors;
};

//abstract interface declaration
class ResourceComm {
public:
    ResourceComm() {};
    virtual ~ResourceComm() {};

    virtual void reset(unsigned layer) = 0;
    virtual void sendInfoMsg(unsigned layer) = 0;

    virtual void newContext(unsigned layer, Matrix &inputTensor_, Matrix &outputTensor_,
                            std::vector<Matrix> *savedTensors_, bool pipeline = false) = 0;
    virtual void newContext(unsigned layer, Matrix &inputTensor_, Matrix &outputTensor_, Matrix &targetTensor_,
                            std::vector<Matrix> *savedTensors_, bool pipeline = false) = 0;

    // For forward-prop.
    virtual void requestForward(unsigned layer, bool lastLayer) = 0;

    virtual void applyVertexForward(unsigned layer, unsigned lambdaId, bool lastLayer) = 0;
    virtual void applyEdgeForward(unsigned layer, unsigned lambdaId, bool lastLayer) = 0;
    virtual void waitResForward(unsigned layer, bool lastLayer) = 0;

    // For backward-prop.
    virtual void requestBackward(unsigned layer, bool lastLayer) = 0;

    virtual void applyVertexBackward(unsigned layer, unsigned lambdaId, bool lastLayer) = 0;
    virtual void applyEdgeBackward(unsigned layer, unsigned lambdaId, bool lastLayer) = 0;
    virtual void waitResBackward(unsigned layer, bool lastLayer) = 0;

    virtual void requestInvoke(unsigned layer, unsigned lambdaId,
      PROP_TYPE prop_dir, bool lastLayer) = 0;
    virtual void waitLambda(unsigned layer, PROP_TYPE prop_dir, bool lastLayer) = 0;

    virtual unsigned getRelaunchCnt() { return 0u; };
    // Send a message to the coordination server to shutdown.
    virtual void sendShutdownMessage() = 0;
};

ResourceComm* createResourceComm(const unsigned& type, CommInfo& commInfo);
void destroyResourceComm(const unsigned& type, ResourceComm *resourceComm);

#endif // __RESOURCE_COMM_HPP__
