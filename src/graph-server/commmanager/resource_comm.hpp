#ifndef __RESOURCE_COMM_HPP__
#define __RESOURCE_COMM_HPP__
#include <string>
#include "../utils/utils.hpp"
#include "../../common/matrix.hpp"
#include <dlfcn.h>

struct CommInfo {
    std::string nodeIp;
    unsigned nodeId;
    unsigned dataserverPort;
    std::string coordserverIp;
    unsigned coordserverPort;
    unsigned numLambdasForward;
    unsigned numLambdasBackward;

    unsigned numNodes;
    std::string wServersFile;
    unsigned weightserverPort;

    unsigned totalLayers; //for weights prefetching
};

//abstract interface declaration
class ResourceComm {
public:
    ResourceComm() {};
    virtual ~ResourceComm() {};
    // For forward-prop.
    virtual void newContextForward(unsigned layer, FeatType *dataBuf, FeatType *zData,
        FeatType *actData, unsigned numLocalVertices, unsigned numFeats,
        unsigned numFeatsNext) = 0;

    virtual void requestForward(unsigned layer, bool lastLayer) = 0;

    virtual void invokeLambdaForward(unsigned layer, unsigned lambdaId, bool lastLayer) = 0;
    virtual void waitLambdaForward(unsigned layer, bool lastLayer) = 0;

    // For backward-prop.
    virtual void newContextBackward(unsigned layer, FeatType *oldGradBuf, FeatType *newGradBuf, std::vector<Matrix> *savedTensors, FeatType *targetBuf,
                                    unsigned numLocalVertices, unsigned inFeatDim, unsigned outFeatDim, unsigned targetDim) = 0;

    virtual void requestBackward(unsigned layer, bool lastLayer) = 0;
    virtual void invokeLambdaBackward(unsigned layer, unsigned lambdaId, bool lastLayer) = 0;
    virtual void waitLambdaBackward(unsigned layer, bool lastLayer) = 0;

    // Send a message to the coordination server to shutdown.
    virtual void sendShutdownMessage() = 0;
};

ResourceComm* createResourceComm(const std::string& type, CommInfo& commInfo);
void destroyResourceComm(const std::string& type, ResourceComm *resourceComm);

#endif // __RESOURCE_COMM_HPP__
