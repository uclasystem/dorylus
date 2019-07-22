#include <vector>
#include <unistd.h>
#include <zmq.hpp>
#include "commmanager.hpp"
#include "../parallel/lock.hpp"
#include "../nodemanager/nodemanager.hpp"


/** Extern class-wide fields. */
unsigned CommManager::nodeId = 0;
unsigned CommManager::numNodes = 0;
std::vector<bool> CommManager::nodesAlive;
unsigned CommManager::numLiveNodes;
unsigned CommManager::dataPort = DATA_PORT;
unsigned CommManager::controlPortStart = CONTROL_PORT_START;
zmq::context_t CommManager::dataContext;                    // Data sockets & locks.
zmq::socket_t *CommManager::dataPublisher = NULL;
zmq::socket_t *CommManager::dataSubscriber = NULL;
Lock CommManager::lockDataPublisher;
Lock CommManager::lockDataSubscriber;
zmq::socket_t **CommManager::controlPublishers = NULL;      // Control sockets & locks.
zmq::socket_t **CommManager::controlSubscribers = NULL;
zmq::context_t CommManager::controlContext;
Lock *CommManager::lockControlPublishers = NULL;
Lock *CommManager::lockControlSubscribers = NULL;


/**
 *
 * Initialize the communication manager. Should be invoked after the node managers have been settled.
 * 
 */
void
CommManager::init() {
    printLog(nodeId, "CommManager starts initialization...");
    numNodes = NodeManager::getNumNodes();
    nodeId = NodeManager::getNodeId();

    Node me = NodeManager::getNode(nodeId);

    // Data publisher & subscribers.
    dataPublisher = new zmq::socket_t(dataContext, ZMQ_PUB);
    assert(dataPublisher->ksetsockopt(ZMQ_SNDHWM, INF_WM));
    assert(dataPublisher->ksetsockopt(ZMQ_RCVHWM, INF_WM));
    char hostPort[50];
    sprintf(hostPort, "tcp://%s:%u", me.ip.c_str(), dataPort);
    assert(dataPublisher->kbind(hostPort));
    dataSubscriber = new zmq::socket_t(dataContext, ZMQ_SUB);
    assert(dataSubscriber->ksetsockopt(ZMQ_SNDHWM, INF_WM));
    assert(dataSubscriber->ksetsockopt(ZMQ_RCVHWM, INF_WM));

    // Connect to all the nodes.
    for (unsigned i = 0; i < numNodes; ++i) {
        Node node = NodeManager::getNode(i);
        char hostPort[50];
        sprintf(hostPort, "tcp://%s:%u", node.ip.c_str(), dataPort);
        dataSubscriber->connect(hostPort);
        nodesAlive.push_back(true);
    }
    numLiveNodes = numNodes;
    dataSubscriber->setsockopt(ZMQ_SUBSCRIBE, NULL, 0);

    lockDataPublisher.init();
    lockDataSubscriber.init();

    // Control publisher & subscribers.
    controlPublishers = new zmq::socket_t*[numNodes];
    controlSubscribers = new zmq::socket_t*[numNodes];
    lockControlPublishers = new Lock[numNodes];
    lockControlSubscribers = new Lock[numNodes];

    // Connect to all the nodes.
    for (unsigned i = 0; i < numNodes; ++i) {
        if(i == nodeId)     // Skip myself.
            continue;

        controlPublishers[i] = new zmq::socket_t(controlContext, ZMQ_PUB);
        assert(controlPublishers[i]->ksetsockopt(ZMQ_SNDHWM, INF_WM));
        assert(controlPublishers[i]->ksetsockopt(ZMQ_RCVHWM, INF_WM));

        char hostPort[50];
        int prt = controlPortStart + i; 
        sprintf(hostPort, "tcp://%s:%d", me.ip.c_str(), prt);
        assert(controlPublishers[i]->kbind(hostPort));

        controlSubscribers[i] = new zmq::socket_t(controlContext, ZMQ_SUB);
        assert(controlSubscribers[i]->ksetsockopt(ZMQ_SNDHWM, INF_WM));
        assert(controlSubscribers[i]->ksetsockopt(ZMQ_RCVHWM, INF_WM));

        Node node = NodeManager::getNode(i);
        sprintf(hostPort, "tcp://%s:%d", node.ip.c_str(), controlPortStart + me.id);
        controlSubscribers[i]->connect(hostPort);
        char tpc = CONTROL_MESSAGE_TOPIC;
        controlSubscribers[i]->setsockopt(ZMQ_SUBSCRIBE, &tpc, 1); 
        
        lockControlPublishers[i].init();
        lockControlSubscribers[i].init();
    }
    
    // Subscribe mutually with everyone.
    bool subscribed[numNodes];  // This arrays says which have been successfully subscribed.
    for (unsigned i = 0; i < numNodes; ++i)
        subscribed[i] = false;
    subscribed[nodeId] = true;
    unsigned remaining = numNodes - 1;
    
    double lastSents[numNodes];
    for (unsigned i = 0; i < numNodes; ++i)
        lastSents[i] = -getTimer() - 1000.0;

    unsigned i = 0;
    while (1) {     // Loop until all have been subscribed.
        if (i == nodeId || subscribed[i]) {     // Skip myself.
            i = (i + 1) % numNodes;
            continue;
        }

        // Send IAMUP.
        if (lastSents[i] + getTimer() > 500) {
            zmq::message_t outMsg(sizeof(ControlMessage));
            ControlMessage cMsg(IAMUP);
            *((ControlMessage *) outMsg.data()) = cMsg;
            controlPublishers[i]->ksend(outMsg);
            lastSents[i] = -getTimer();
        }

        // Received a message from the same node.
        zmq::message_t inMsg;
        if (controlSubscribers[i]->krecv(&inMsg, ZMQ_DONTWAIT)) {

            // Received IAMUP from that node.
            ControlMessage cMsg = *((ControlMessage *) inMsg.data());
            if(cMsg.messageType == IAMUP) {

                // Send ISEEYOU.
                {
                    zmq::message_t outAckMsg(sizeof(ControlMessage));
                    ControlMessage cAckMsg(ISEEYOUUP);
                    *((ControlMessage *) outAckMsg.data()) = cAckMsg;
                    controlPublishers[i]->ksend(outAckMsg);
                }

                // Loop until receiving ISEEYOU from that node.
                while (1) {
                    zmq::message_t inAckMsg;
                    if (controlSubscribers[i]->krecv(&inAckMsg, ZMQ_DONTWAIT)) {
                        cMsg = *((ControlMessage *) inAckMsg.data());
                        if (cMsg.messageType == ISEEYOUUP) {
                            zmq::message_t outAckMsg(sizeof(ControlMessage));
                            ControlMessage cAckMsg(ISEEYOUUP);
                            *((ControlMessage *) outAckMsg.data()) = cAckMsg;  
                            controlPublishers[i]->ksend(outAckMsg);

                            subscribed[i] = true;
                            --remaining;
                            break;
                        }
                    } else 
                        break;
                }

            // Received ISEEYOU from that node.
            } else if (cMsg.messageType == ISEEYOUUP) {

                // Send ISEEYOU back.
                zmq::message_t outAckMsg(sizeof(ControlMessage));
                ControlMessage cAckMsg(ISEEYOUUP);
                *((ControlMessage *) outAckMsg.data()) = cAckMsg;
                controlPublishers[i]->ksend(outAckMsg);

                subscribed[i] = true;
                --remaining;
            }
        }

        // If all nodes have been subscribed, subsription success; else go check the next node.
        if (remaining == 0)
            break;
        else
            i = (i + 1) % numNodes;
    }

    flushControl();
    printLog(nodeId, "CommManager initialization complete.");
}


/**
 *
 * Destroy the communication manager.
 * 
 */
void
CommManager::destroy() {

    flushControl();
    flushData();

    // Data publisher & subscriber.
    dataPublisher->close();
    dataSubscriber->close();

    delete dataPublisher;
    delete dataSubscriber;

    lockDataPublisher.destroy();
    lockDataSubscriber.destroy();

    // Control publishers & subscribers.
    for (unsigned i = 0; i < numNodes; ++i) {
        if(i == nodeId)     // Skip myself.
            continue;

        controlPublishers[i]->close();
        controlSubscribers[i]->close();
        delete controlPublishers[i];
        delete controlSubscribers[i];
    
        lockControlPublishers[i].destroy();
        lockControlSubscribers[i].destroy();
    }

    delete[] controlPublishers;
    delete[] controlSubscribers;
    delete[] lockControlPublishers;
    delete[] lockControlSubscribers;

    dataContext.close();
    controlContext.close();
}


/**
 *
 * Push a value on a certain topic to all the nodes (including myself).
 * 
 */
void
CommManager::dataPushOut(IdType topic, void *value, unsigned valSize) {
    zmq::message_t outMsg(sizeof(IdType) + valSize);
    *((IdType *) outMsg.data()) = topic;
    memcpy((void *)(((char *) outMsg.data()) + sizeof(IdType)), value, valSize);

    lockDataPublisher.lock();
    dataPublisher->ksend(outMsg, ZMQ_DONTWAIT); 
    lockDataPublisher.unlock();
}


/**
 *
 * Pull a value on a certain topic in.
 * 
 */
bool
CommManager::dataPullIn(IdType &topic, std::vector<FeatType>& value) {
    zmq::message_t inMsg;

    lockDataSubscriber.lock();
    bool ret = dataSubscriber->krecv(&inMsg, ZMQ_DONTWAIT);
    lockDataSubscriber.unlock();

    if (!ret)
        return false;

    int32_t dataSize = inMsg.size() - sizeof(IdType);
    int32_t numberOfFeatures = dataSize / sizeof(FeatType);
    value.resize(numberOfFeatures);

    memcpy(&topic, inMsg.data(), sizeof(IdType));
    memcpy(value.data(), ((char *)inMsg.data() + sizeof(IdType)), dataSize);

    return true;
}


/**
 *
 * Push a value to a specific node (cannot be myself).
 * 
 */
void
CommManager::controlPushOut(unsigned to, void *value, unsigned valSize) {
    assert(to >= 0 && to < numNodes);
    assert(to != nodeId); 
    zmq::message_t outMsg(sizeof(ControlMessage) + valSize);
    *((ControlMessage *) outMsg.data()) = ControlMessage(APPMSG);
    memcpy((void *)(((char *) outMsg.data()) + sizeof(ControlMessage)), value, valSize);

    lockControlPublishers[to].lock();
    assert(controlPublishers[to]->ksend(outMsg, ZMQ_DONTWAIT));
    lockControlPublishers[to].unlock();
}


/**
 *
 * Pull a value in from a specific node (cannot be myself).
 * 
 */
bool
CommManager::controlPullIn(unsigned from, void *value, unsigned valSize) {
    assert(from >= 0 && from < numNodes);
    assert(from != nodeId);
    zmq::message_t inMsg;

    lockControlSubscribers[from].lock();
    bool ret = controlSubscribers[from]->krecv(&inMsg, ZMQ_DONTWAIT);
    lockControlSubscribers[from].unlock();

    if (!ret)
        return false;

    ControlMessage cM = *((ControlMessage *) inMsg.data());
    assert(cM.messageType == APPMSG);

    memcpy(value, ((void *)((char *)inMsg.data() + sizeof(ControlMessage))), valSize);
    return true;
}


/**
 *
 * Flush the data communication pipe between myself and all living nodes.
 * 
 */
void
CommManager::flushData() {
    lockDataPublisher.lock();
    lockDataSubscriber.lock();

    zmq::message_t outMsg(sizeof(IdType));
    *((IdType*) outMsg.data()) = NULL_CHAR;
   
    dataPublisher->ksend(outMsg);

    unsigned rem = numLiveNodes;

    while (rem > 0) {
        zmq::message_t inMsg;
        assert(dataSubscriber->krecv(&inMsg));
        IdType idx = *((IdType *) inMsg.data());
        if (idx == NULL_CHAR)
            --rem;
    }

    lockDataSubscriber.unlock();
    lockDataPublisher.unlock();
}


/**
 *
 * Flush the control communication pipe between myself and all living nodes.
 * 
 */
void CommManager::flushControl() {
    for (unsigned i = 0; i < numNodes; ++i) {
        if((i == nodeId) || (nodesAlive[i] == false))   // Skip myself and died nodes.
            continue;

        lockControlPublishers[i].lock();
        lockControlSubscribers[i].lock();

        zmq::message_t outAckMsg(sizeof(ControlMessage));
        *((ControlMessage *) outAckMsg.data()) = ControlMessage();
        controlPublishers[i]->ksend(outAckMsg);

        while(1) {
            zmq::message_t inMsg;
            if (controlSubscribers[i]->krecv(&inMsg, ZMQ_DONTWAIT)) {
                ControlMessage cMsg = *((ControlMessage *) inMsg.data());
                if (cMsg.messageType == NONE)
                    break;
            }
        }

        lockControlSubscribers[i].unlock();
        lockControlPublishers[i].unlock();
    }
}


/**
 *
 * Sentence a living node to death.
 * 
 */
void CommManager::nodeDie(unsigned nId) {
    assert(nodesAlive[nId]);
    nodesAlive[nId] = false;
    --numLiveNodes;
}
