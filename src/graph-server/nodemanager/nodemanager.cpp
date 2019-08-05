#include <fstream>
#include <cassert>
#include <cstring>
#include <boost/algorithm/string/trim.hpp>
#include "nodemanager.hpp"
#include "../utils/utils.hpp"


/** Extern class-wide fields. */
Node NodeManager::me;
std::vector<Node> NodeManager::allNodes;
bool NodeManager::inBarrier = false;


/**
 *
 * Initialize the communication manager. Should be done before initializing the CommManager.
 * Node manager can only be invoked by a single thread (typically the master thread).
 * 
 */
void
NodeManager::init(const std::string dshMachinesFile) {
    printLog(404, "NodeManager starts initialization...\n");

    getIP(&me.ip);
    getPubIP(me.pubip);

    parseNodeConfig(dshMachinesFile);

    printLog(me.id, "NodeManager initialization complete.\n");
}


/**
 *
 * Public API for a node to hit a global barrier.
 * 
 */
void NodeManager::barrier() {
    inBarrier = true;

    // TODO...

    inBarrier = false;
}


/**
 *
 * Destroy the node manager.
 * 
 */
void
NodeManager::destroy() {
    barrier();
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
NodeManager::getNodeId() {
    return me.id; 
}

bool
NodeManager::amIMaster() {
    return me.master;
}

unsigned
NodeManager::getMasterId() {
    return masterId; 
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
