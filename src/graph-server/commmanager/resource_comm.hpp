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

    FuncPtr scatterFunc;
};

//abstract interface declaration
class ResourceComm {
public:
    ResourceComm() {};
    virtual ~ResourceComm() {};
    // For forward-prop.
    virtual void newContextForward(unsigned layer, FeatType *dataBuf, FeatType *zData,
        FeatType *actData, unsigned numLocalVertices, unsigned numFeats,
        unsigned numFeatsNext, bool pipeline = false) = 0;

    virtual void requestForward(unsigned layer, bool lastLayer) = 0;

    virtual void invokeLambdaForward(unsigned layer, unsigned lambdaId, bool lastLayer) = 0;
    virtual void waitLambdaForward(unsigned layer, bool lastLayer) = 0;

    // For backward-prop.
    virtual void newContextBackward(unsigned layer, FeatType *oldGradBuf, FeatType *newGradBuf, std::vector<Matrix> *savedTensors, FeatType *targetBuf,
                                    unsigned numLocalVertices, unsigned inFeatDim, unsigned outFeatDim, unsigned targetDim) = 0;

    virtual void requestBackward(unsigned layer, bool lastLayer) = 0;
    virtual void invokeLambdaBackward(unsigned layer, unsigned lambdaId, bool lastLayer) = 0;
    virtual void waitLambdaBackward(unsigned layer, bool lastLayer) = 0;

    virtual void sendShutdownMessage() = 0;
};

ResourceComm* createResourceComm(const unsigned& type, CommInfo& commInfo);
void destroyResourceComm(const unsigned& type, ResourceComm *resourceComm);

#endif // __RESOURCE_COMM_HPP__
