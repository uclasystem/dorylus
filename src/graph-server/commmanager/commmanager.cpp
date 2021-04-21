#include <unistd.h>
#include "commmanager.hpp"


/**
 *
 * Initialize the communication manager. Should be invoked after the node managers have been settled.
 *
 */
void
CommManager::init(NodeManager& nodeManager, unsigned ctxThds) {
    printLog(nodeId, "CommManager starts initialization...");
    numNodes = nodeManager.getNumNodes();
    numWorkers = nodeManager.getNumWorkers();
    nodeId = nodeManager.getMyNodeId();
    Node me = nodeManager.getNode(nodeId);
    localId = me.localId;

    if (nodeManager.standAloneMode() == true) {
        printLog(nodeId, "CommManager initialization complete.");
        return;
    }

    // Set data context threads for high scatter bandwidth
    zmq_ctx_set((void *)dataContext, ZMQ_IO_THREADS, ctxThds);

    // Data publisher & subscriber.
    dataPublisher = new zmq::socket_t(dataContext, ZMQ_PUB);
    dataPublisher->setsockopt(ZMQ_SNDHWM, 0);       // Set no limit on number of message queueing.
    dataPublisher->setsockopt(ZMQ_RCVHWM, 0);
    char hostPort[50];
    sprintf(hostPort, "tcp://%s:%u", me.ip.c_str(), dataPort + localId);
    dataPublisher->bind(hostPort);

    dataSubscriber = new zmq::socket_t(dataContext, ZMQ_SUB);
    dataSubscriber->setsockopt(ZMQ_SNDHWM, 0);
    dataSubscriber->setsockopt(ZMQ_RCVHWM, 0);
    char filter[9]; // filter for data subscriber
    sprintf(filter, "%8X", nodeId);
    dataSubscriber->setsockopt(ZMQ_SUBSCRIBE, filter, 8);
    dataSubscriber->setsockopt(ZMQ_SUBSCRIBE, "FFFFFFFF", 8);
    for (unsigned i = 0; i < numNodes; ++i) {
        Node node = nodeManager.getNode(i);
        if (node.ip == me.ip) {
            localNodeIdMin = std::min(localNodeIdMin, node.id);
            localNodeIdMax = std::max(localNodeIdMax, node.id);
        }
        char hostPort[50];
        sprintf(hostPort, "tcp://%s:%u", node.ip.c_str(), dataPort + node.localId);
        dataSubscriber->connect(hostPort);
    }

    printLog(nodeId, "Boundaries for my IP %s: (%u,%u)", me.ip.c_str(), localNodeIdMin, localNodeIdMax);

    if (numWorkers > 1) {
        ipcSockets = new zmq::socket_t*[numWorkers];
        lockIpcSockets = new Lock[numWorkers];
        for (int wid = 0; wid < numWorkers; ++wid) {
            lockIpcSockets[wid].init();
            std::string address = "ipc:///home/ubuntu/feeds/" + std::to_string(wid);
            if (wid == me.localId) {
                ipcSockets[wid] = new zmq::socket_t(dataContext, ZMQ_DEALER);
                ipcSockets[wid]->bind(address.c_str());
            }
            else {
                ipcSockets[wid] = new zmq::socket_t(dataContext, ZMQ_DEALER);
                ipcSockets[wid]->connect(address.c_str());
            }
        }
    }
    sleep_ms(5000);

    lockDataPublisher.init();
    lockDataSubscriber.init();

    flushData();
    //flushControl();
    printLog(nodeId, "CommManager initialization complete.");
}


/**
 *
 * Destroy the communication manager.
 *
 */
void
CommManager::destroy() {

    if (numNodes == 0) return;

//    flushControl();
    flushData();

    // Data publisher & subscriber.
    dataPublisher->close();
    dataSubscriber->close();

    delete dataPublisher;
    delete dataSubscriber;

    lockDataPublisher.destroy();
    lockDataSubscriber.destroy();

//    // Control publishers & subscribers.
//    for (unsigned i = 0; i < numNodes; ++i) {
//        if (i == nodeId)        // Skip myself.
//            continue;
//
//        controlPublishers[i]->close();
//        controlSubscribers[i]->close();
//        delete controlPublishers[i];
//        delete controlSubscribers[i];
//
//        lockControlPublishers[i].destroy();
//        lockControlSubscribers[i].destroy();
//    }
//
//    delete[] controlPublishers;
//    delete[] controlSubscribers;
//    delete[] lockControlPublishers;
//    delete[] lockControlSubscribers;
}

void
CommManager::rawMsgPushOut(unsigned receiver, zmq::message_t &msg) {
    if (numNodes == 0) return;

//    if (receiver >= localNodeIdMin && receiver <= localNodeIdMax) {
//        unsigned localNodeId = receiver - localNodeIdMin;
//        lockIpcSockets[localNodeId].lock();
//        ipcSockets[localNodeId]->ksend(msg, ZMQ_DONTWAIT);
//        lockIpcSockets[localNodeId].unlock();
//    } else {
        lockDataPublisher.lock();
        dataPublisher->ksend(msg, ZMQ_DONTWAIT);
        lockDataPublisher.unlock();
//    }
}

/**
 *
 * Push a value on a certain topic to all the nodes (including myself).
 *
 */
void
CommManager::dataPushOut(unsigned receiver, unsigned sender, unsigned topic, void *value, unsigned valSize) {
    if (numNodes == 0) return;

    zmq::message_t outMsg(sizeof(char) * 8 + sizeof(unsigned) + sizeof(unsigned) + valSize);
    char *msgPtr = (char *)outMsg.data();
    sprintf(msgPtr, "%8X", receiver);
    msgPtr += 8;
    *((unsigned *)msgPtr) = sender;
    msgPtr += sizeof(unsigned);
    *((unsigned *)msgPtr) = topic;
    msgPtr += sizeof(unsigned);

    if (valSize > 0)
        memcpy((void *)msgPtr, value, valSize);

//    if (receiver >= localNodeIdMin && receiver <= localNodeIdMax) {
//        unsigned localNodeId = receiver - localNodeIdMin;
//        lockIpcSockets[localNodeId].lock();
//        ipcSockets[localNodeId]->ksend(outMsg, ZMQ_DONTWAIT);
//        lockIpcSockets[localNodeId].unlock();
//    } else {
        lockDataPublisher.lock();
        dataPublisher->ksend(outMsg, ZMQ_DONTWAIT);
        lockDataPublisher.unlock();
//    }
}


/**
 *
 * Pull a value on a certain topic in.
 *
 */
bool
CommManager::dataPullIn(unsigned *sender, unsigned *topic, void *value, unsigned maxValSize) {
    if (numNodes == 0) return true;

    zmq::message_t inMsg;

    lockDataSubscriber.lock();
    bool ret = dataSubscriber->krecv(&inMsg, ZMQ_DONTWAIT);
    lockDataSubscriber.unlock();

//    bool ipcRet = false;
//    if (!ret) {
//        for (int u = 0; u < numWorkers; ++u) {
//            int currIndex = (pollSocketIndex + u) % numWorkers;
//            if (currIndex == ((int)localId)) continue;
//            lockIpcSockets[currIndex].lock();
//            ipcRet = ipcSockets[currIndex]->krecv(&inMsg, ZMQ_DONTWAIT);
//            lockIpcSockets[currIndex].unlock();
//            ret = ipcRet;
//            if (ret) {
//                pollSocketIndex = currIndex;
//                break;
//            }
//        }
//    }
    if (!ret) return false;
//    if (ipcRet) printLog(nodeId, "GOT SOMETHING FROM IPC SOCKET");

    unsigned valSize = inMsg.size() - sizeof(char) * 8 - sizeof(unsigned) - sizeof(unsigned);
    assert(valSize <= maxValSize);
    char *msgPtr = (char *)inMsg.data();
    msgPtr += 8;
    memcpy(sender, (unsigned *)msgPtr, sizeof(unsigned));
    msgPtr += sizeof(unsigned);
    memcpy(topic, msgPtr, sizeof(unsigned));
    msgPtr += sizeof(unsigned);
    memcpy(value, msgPtr, valSize);

    return true;
}


/**
 *
 * Push a value to a specific node (cannot be myself).
 *
 */
void
CommManager::controlPushOut(unsigned to, void *value, unsigned valSize) {
    if (numNodes == 0) return;

    assert(to >= 0 && to < numNodes);
    assert(to != nodeId);
    zmq::message_t outMsg(sizeof(ControlMessage) + valSize);
    *((ControlMessage *) outMsg.data()) = ControlMessage(APPMSG);

    if (valSize > 0)
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
CommManager::controlPullIn(unsigned from, void *value, unsigned maxValSize) {

    if (numNodes == 0) return true;

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

    assert(inMsg.size() - sizeof(ControlMessage) <= maxValSize);
    memcpy(value, ((char *) inMsg.data() + sizeof(ControlMessage)), inMsg.size() - sizeof(ControlMessage));

    return true;
}


///////////////////////////////////////////////////////////////
// Below are private function for the communication manager. //
///////////////////////////////////////////////////////////////


/**
 *
 * Flush the data communication pipe between myself and all living nodes.
 *
 */
void
CommManager::flushData() {
    lockDataPublisher.lock();
    lockDataSubscriber.lock();

    zmq::message_t outMsg(sizeof(char) * 8 + sizeof(unsigned));
    char *msgPtr = (char *)(outMsg.data());
    sprintf(msgPtr, "FFFFFFFF");
    msgPtr += 8;
    *(unsigned *)msgPtr = NULL_CHAR;
    dataPublisher->ksend(outMsg);

    unsigned rem = numNodes;

    while (rem > 0) {
        zmq::message_t inMsg;
        dataSubscriber->recv(&inMsg);
        char *msgPtr = (char *)inMsg.data();
        msgPtr += 8;
        unsigned idx = *((unsigned *)msgPtr);
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

        if (i == nodeId)    // Skip myself.
            continue;

        lockControlPublishers[i].lock();
        lockControlSubscribers[i].lock();

        zmq::message_t outAckMsg(sizeof(ControlMessage));
        *((ControlMessage *) outAckMsg.data()) = ControlMessage();
        controlPublishers[i]->ksend(outAckMsg);

        while (1) {
            zmq::message_t inMsg;
            if (controlSubscribers[i]->krecv(&inMsg, ZMQ_DONTWAIT)) {
                ControlMessage cMsg = *((ControlMessage *) inMsg.data());
                if (cMsg.messageType == CTRLNONE)
                    break;
            }
        }

        lockControlSubscribers[i].unlock();
        lockControlPublishers[i].unlock();
    }
}
