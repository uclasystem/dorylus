#ifndef __RESOURCE_COMM_HPP__
#define __RESOURCE_COMM_HPP__
#include <string>
#include "../utils/utils.hpp"
#include "../../common/matrix.hpp"
#include <dlfcn.h>

struct CommInfo{
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
};

//abstract interface declaration
class ResourceComm
{
public:
	ResourceComm(){};
	virtual void setTrainValidationSplit(float trainPortion, unsigned numLocalVertices)=0;

    // For forward-prop.
    virtual void newContextForward(FeatType *dataBuf, FeatType *zData,
        FeatType *actData, unsigned numLocalVertices, unsigned numFeats,
        unsigned numFeatsNext, bool eval)=0;

    virtual void requestForward(unsigned layer,bool lastLayer)=0;

    virtual void invokeLambdaForward(unsigned layer, unsigned lambdaId, bool lastLayer)=0;
    virtual void waitLambdaForward()=0;

    // For backward-prop.
    virtual void newContextBackward(FeatType **zBufs, FeatType **actBufs, FeatType *targetBuf,
                            unsigned numLocalVertices, std::vector<unsigned> layerConfig)=0;
    virtual void newContextBackward(FeatType *oldGradBuf, FeatType *newGradBuf, std::vector<Matrix> *savedTensors, FeatType *targetBuf,
                                    unsigned numLocalVertices, unsigned inFeatDim, unsigned outFeatDim, unsigned targetDim) = 0;

    virtual void requestBackward(unsigned numLayers_, bool lastLayer)=0;

    // Send a message to the coordination server to shutdown.
    virtual void sendShutdownMessage()=0;

	~ResourceComm(){};

};


ResourceComm* createResourceComm(const std::string& type, CommInfo& commInfo);



#endif // __RESOURCE_COMM_HPP__