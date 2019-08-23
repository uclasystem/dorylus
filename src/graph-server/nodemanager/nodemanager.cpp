#include <fstream>
#include <cassert>
#include <cstring>
#include <chrono>
#include <thread>
#include <boost/algorithm/string/trim.hpp>
#include "nodemanager.hpp"
#include "../utils/utils.hpp"


/** Extern class-wide fields. */
Node NodeManager::me;
std::vector<Node> NodeManager::allNodes;
bool NodeManager::inBarrier = false;
zmq::context_t NodeManager::nodeContext;                // Node barriering sockets & locks.
zmq::socket_t *NodeManager::nodePublisher = NULL;
zmq::socket_t *NodeManager::nodeSubscriber = NULL;
unsigned NodeManager::nodePort;


/**
 *
 * Initialize the communication manager. Should be done before initializing the CommManager.
 * Node manager is NOT thread safe - it can only be invoked by a single computation thread (typically the master thread).
 * 
 */
void
NodeManager::init(std::string dshMachinesFile, std::string myPrIpFile, std::string myPubIpFile) {
    printLog(me.id, "NodeManager starts initialization...");
    getPrIP(myPrIpFile, me.ip);
    getPubIP(myPubIpFile, me.pubip);
    parseNodeConfig(dshMachinesFile);
    allNodes[me.id].pubip = me.pubip;       // Set my node struct's pubip, in case CommManager uses it.
    printLog(me.id, "Private IP: %s, Public IP: %s", me.ip.c_str(), me.pubip.c_str());

    // Initialize node barriering sockets.
    nodePublisher = new zmq::socket_t(nodeContext, ZMQ_PUB);
    nodePublisher->setsockopt(ZMQ_SNDHWM, 0);       // Set no limit on number of message queueing.
    nodePublisher->setsockopt(ZMQ_RCVHWM, 0);
    char hostPort[50];
    // sprintf(hostPort, "tcp://%s:%u", me.ip.c_str(), nodePort);
    sprintf(hostPort, "tcp://%s:%u", "*", nodePort); //GPU
    printLog(me.id, "Binding to %s",hostPort);//GPU
    nodePublisher->bind(hostPort);

    nodeSubscriber = new zmq::socket_t(nodeContext, ZMQ_SUB);
    nodeSubscriber->setsockopt(ZMQ_SNDHWM, 0);
    nodeSubscriber->setsockopt(ZMQ_RCVHWM, 0);
    for (Node& node : allNodes) {
        char hostPort[50];
        sprintf(hostPort, "tcp://%s:%u", node.ip.c_str(), nodePort);
        nodeSubscriber->connect(hostPort);
    }
    nodeSubscriber->setsockopt(ZMQ_SUBSCRIBE, NULL, 0);

    // The master node keeps polling on incoming respond WORKERUP messages until everyone else
    // has finished initialization.
    if (me.master) {
        unsigned remaining = getNumNodes() - 1;

        // Keeps polling until all workers' respond processed.
        while (remaining > 0) {

            // Send MASTERUP.
            zmq::message_t outMsg(sizeof(NodeMessage));
            NodeMessage nMsg(MASTERUP);
            *((NodeMessage *) outMsg.data()) = nMsg;
            nodePublisher->ksend(outMsg);

            // Sleep for 0.5 sec before checking the responds, to avoid clobbing the sockets.
            std::this_thread::sleep_for(std::chrono::milliseconds(500));

            // Receive all WORKERUP responds in queue (don't do block recv here).
            zmq::message_t inMsg;
            while (nodeSubscriber->krecv(&inMsg, ZMQ_DONTWAIT)) {
                NodeMessage nMsg = *((NodeMessage *) inMsg.data());
                if (nMsg.messageType == WORKERUP){
                    --remaining;
                }
            }
        }

        // Signal workers to move on.
        zmq::message_t outMsg(sizeof(NodeMessage));
        NodeMessage nMsg(INITDONE);
        *((NodeMessage *) outMsg.data()) = nMsg;
        nodePublisher->send(outMsg);

    // Worker nodes, when received master's MASTERUP message, respond a WORKERUP message.
    } else {

        // Wait for master's polling request.
        zmq::message_t inMsg;

        while (nodeSubscriber->recv(&inMsg)) {
            NodeMessage nMsg = *((NodeMessage *) inMsg.data());
            if (nMsg.messageType == MASTERUP)
                break;
        }

        // Send back respond message to master.
        zmq::message_t outMsg(sizeof(NodeMessage));
        NodeMessage nMsg(WORKERUP);
        *((NodeMessage *) outMsg.data()) = nMsg;
        nodePublisher->send(outMsg);

        // Wait for master's signal to move forward.
        while (nodeSubscriber->recv(&inMsg)) {
            NodeMessage nMsg = *((NodeMessage *) inMsg.data());
            if (nMsg.messageType == INITDONE)
                break;
        }
    }

    printLog(me.id, "NodeManager initialization complete.");
}


/**
 *
 * Public API for a node to hit a global barrier.
 * 
 */
void
NodeManager::barrier() {
    inBarrier = true;
    printLog(me.id, "Hits on a global barrier |xxx|...");

    // Send BARRIER message.
    zmq::message_t outMsg(sizeof(NodeMessage));
    NodeMessage nMsg(BARRIER);
    *((NodeMessage *) outMsg.data()) = nMsg;
    nodePublisher->send(outMsg);

    // Keeps receiving BARRIER messages until heard from all (including self).
    unsigned remaining = allNodes.size();
    zmq::message_t inMsg;
    while (remaining > 0) {
        nodeSubscriber->recv(&inMsg);
        NodeMessage nMsg = *((NodeMessage *) inMsg.data());
        if (nMsg.messageType == BARRIER)
            --remaining;
    }

    // No redundant messages sent, so there should be no remaining messages in flight or in someone's
    // message queue after leaving the global barrier. Thus, we do not need to flush.

    inBarrier = false;
    printLog(me.id, "Left that global barrier |xxx|.");
}


/**
 *
 * Destroy the node manager.
 * 
 */
void
NodeManager::destroy() {
    barrier();

    nodePublisher->close();
    nodeSubscriber->close();

    delete nodePublisher;
    delete nodeSubscriber;
}


/**
 *
 * General public utilities.
 * 
 */
Node&
NodeManager::getNode(unsigned i) {
    return allNodes[i];
}

unsigned
NodeManager::getNumNodes() {
    return allNodes.size();
}

unsigned
NodeManager::getMyNodeId() {
    return me.id;
}

bool
NodeManager::amIMaster() {
    return me.master;
}


///////////////////////////////////////////////////////
// Below are private functions for the node manager. //
///////////////////////////////////////////////////////


/**
 *
 * Parse the node config file.
 * 
 */
void
NodeManager::parseNodeConfig(const std::string dshMachinesFile) {
    std::ifstream inFile(dshMachinesFile);
    std::string line;
    while (std::getline(inFile, line)) {
        boost::algorithm::trim(line);
        if (line.length() > 0) {

            // Get id and private ip of the machine in the line. Machine at line i (starting from 0) will have nodeId = i.
            unsigned id = allNodes.size();
            std::string ip = line.substr(line.find('@') + 1);

            // It's me!
            if (ip == me.ip) {
                me.id = id;
                me.master = (me.id == MASTER_NODEID);
            }

            allNodes.push_back(Node(id, &ip, (id == MASTER_NODEID)));
        }
    }
}
