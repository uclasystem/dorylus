#include <fstream>
#include <cassert>
#include <cstring>
#include "nodemanager.hpp"
#include "zkinterface.hpp"
#include "../utils/utils.hpp"


/** Extern class-wide fields. */
Node NodeManager::me;
unsigned NodeManager::masterId = 0;
std::vector<Node> NodeManager::allNodes;
unsigned NodeManager::numLiveNodes;
Lock NodeManager::lockWaiter;
Cond NodeManager::condWaiter;
Phase NodeManager::phase = REGISTERING;
Lock NodeManager::lockPhase; 
std::map<std::string, BarrierContext> NodeManager::appBarriers;
Lock NodeManager::lockAppBarriers;
void (*NodeManager::ndFunc)(unsigned) = NULL;
std::string NodeManager::zooHostPort;
std::atomic<bool> NodeManager::forceRelease = ATOMIC_VAR_INIT(false);
std::atomic<bool> NodeManager::inBarrier = ATOMIC_VAR_INIT(false);


/**
 *
 * Initialize the communication manager. Should be done before initializing the CommManager.
 * 
 */
void
NodeManager::init(const char *zooHostFile, const char *hostFile) {
    getIP(&me.ip);

    // Parse the config files.
    parseZooConfig(zooHostFile);
    parseNodeConfig(hostFile);

    printLog(me.id, "NodeManager starts initialization... (ZK host port = %s)\n", zooHostPort.c_str());
    assert(ZKInterface::init(zooHostPort.c_str()));

    // Initialize synchronization utilities.
    lockWaiter.init();
    condWaiter.init(lockWaiter);
    lockPhase.init();
    lockAppBarriers.init();

    // Master node registers the barriers for others.
    if (me.master) {
        ZKInterface::recursiveDeleteZKNode(ZK_ROOT_NODE);
        createNode(ZK_ROOT_NODE, false, true, &createCB);
        createNode(ZK_MASTER_NODE, false, true, &createCB);
        createNode(ZK_APPBARRIER_NODE, false, false, &createCB);

        // Master creates a barrier so that everyone must hit here before registration.
        createNode(ZK_BARRIER_NODE, false, false, &createCB);

        std::string subNode = ZK_MASTER_NODE;
        subNode += "/";
        subNode += me.name;
        createNode(subNode.c_str(), true, false, &createCB);

        // Block till everyone gets registered.
        waitForAllNodes();

        // Master deletes this barrier, allowing others to proceed.
        ZKInterface::deleteZKNode(ZK_BARRIER_NODE);

    // Worker nodes wait until the barriers have been registered.
    } else {

        // Everyone hits here before registration.
        hitBarrier();

        std::string subNode = ZK_MASTER_NODE;
        subNode += "/";
        subNode += me.name;
        ZKInterface::createZKNode(subNode.c_str(), true, false, &createCB);

        // Block until master deletes the barrier.
        leaveBarrier();

        watchAllNodes();

        phase = REGISTERED;
    }

    printLog(me.id, "NodeManager initialization complete.\n");
}


/**
 *
 * Public API for a node to hit a global barrier.
 * 
 */
void NodeManager::barrier(const char* bar) {
    inBarrier = true;

    BarrierContext bContext(bar);
    std::string barName = bContext.path;

    lockAppBarriers.lock();
    appBarriers[barName] = bContext;    // Overwrite barrier.
    lockAppBarriers.unlock();

    // Master node is responsible for creating the barrier, and other nodes wait until master has created it. Then
    // the nodes block until all the nodes are trying to leave the barrier.
    //
    // Master's job:
    //      1. Create ZK_MASTER_NODE
    //      2. Create ZK_BARRIER_NODE
    //      3. Create ZK_MASTER_NODE/<name>
    //      4. Wait for all nodes to be created
    //      5. Delete ZK_BARRIER_NODE
    //
    if (me.master) {
        createNode(barName.c_str(), false, true, &createCB);

        lockWaiter.lock();
        
        struct String_vector children;
        ZKInterface::getZKNodeChildren(barName.c_str(), barrierCB, &children);

        if ((unsigned) children.count != numLiveNodes - 1) {
            condWaiter.wait();
            lockWaiter.unlock();
        } else {
            lockWaiter.unlock();
            printLog(me.id, "Everyone reached the barrier (head start).\n", children.count);
            lockAppBarriers.lock();
            appBarriers[barName].ignoreCB = true; 
            lockAppBarriers.unlock();
        }

        ZKInterface::freeZKStringVector(&children);
        ZKInterface::recursiveDeleteZKNode(barName.c_str());

    // Non-master node, so first wait on hitting the barrier then wait on leaving it.
    //
    // Non-master's job:
    //      1. Wait for ZK_BARRIER_NODE to be created
    //      2. Create ZK_MASTER_NODE/<name>
    //      3. Wait for ZK_BARRIER_NODE to be destroyed
    //
    } else {
        printLog(me.id, "Barrier %s: Hitting on it...\n", barName.c_str());
        
        lockWaiter.lock();
        if (!ZKInterface::checkZKExists(barName.c_str(), checkBarrier))
            condWaiter.wait();
        lockWaiter.unlock();

        printLog(me.id, "Barrier %s: Entered.\n", barName.c_str());

        if (!forceRelease) {
            std::string subNode = barName;
            subNode += "/";
            subNode += me.name;
            createNode(subNode.c_str(), true, false, &createCB);
        }

        printLog(me.id, "Barrier %s: Waiting to leave...\n", barName.c_str());

        if (!forceRelease) {
            lockWaiter.lock();
            if (ZKInterface::checkZKExists(barName.c_str(), checkBarrier))
                condWaiter.wait();
            lockWaiter.unlock();
        }
        printLog(me.id, "Barrier %s: Passed through!\n", barName.c_str());

        if (me.master)  // This is possible because master itself died and I got re-elected as new master. 
            ZKInterface::recursiveDeleteZKNode(barName.c_str());
    }

    inBarrier = false;
    forceRelease = false;
}


/**
 *
 * Destroy the node manager.
 * 
 */
void
NodeManager::destroy() {
    phase = UNREGISTERING;
    barrier(DESTROY_BARRIER); 
    if (me.master)
        ZKInterface::recursiveDeleteZKNode(ZK_ROOT_NODE);
}


/**
 *
 * General public utilities.
 * 
 */
std::vector<Node> *
NodeManager::getAllNodes() {
    return &allNodes;
}

Node
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


/**
 *
 * Get node id from its name string.
 * 
 */
unsigned
NodeManager::getNodeId(std::string& nodeName) {
    for (unsigned i = 0; i < allNodes.size(); ++i) {
        if (allNodes[i].name == nodeName)
            return i;
    }
    return allNodes.size();
}


/**
 *
 * Get node name string from id number.
 * 
 */
std::string
NodeManager::getNodeName(const char *path) {
    char *cptr = const_cast<char *>(path) + strlen(path) - 1;
    assert(*cptr != '/');
    while (cptr != path) {
        --cptr;
        if (*cptr == '/') {
            ++cptr;
            break;
        }
    }
    return std::string(cptr);
}


/**
 *
 * Release all nodes in given barrier.
 * 
 */
void
NodeManager::releaseAll(const char *bar) {
    assert(phase == PROCESSING);

    BarrierContext bContext(bar);
    std::string barName = bContext.path;

    lockAppBarriers.lock();
    if (appBarriers.find(barName.c_str()) != appBarriers.end())
        appBarriers[barName.c_str()].ignoreCB = true;
    lockAppBarriers.unlock();

    lockWaiter.lock();
    forceRelease = true;
    condWaiter.signal();
    lockWaiter.unlock();

    forceRelease = inBarrier ? true : false;
}


/**
 *
 * Register a new function to be executed when a node goes down.
 * 
 */
void
NodeManager::registerNodeDownFunc(void (*func)(unsigned)) {
    ndFunc = func;
}


/////////////////////////////////////////////////
// Below are private functions for the engine. //
/////////////////////////////////////////////////


/**
 *
 * Parse the ZooKeeper config file.
 * 
 */
void
NodeManager::parseZooConfig(const char *zooHostFile) {
    zooHostPort = "";
    bool first = true;
    std::ifstream inFile(zooHostFile);
    std::string host, port;
    while (inFile >> host >> port) {
        if (!first)
            zooHostPort += ",";
        zooHostPort += host + ":" + port;
        first = false;
    }
}


/**
 *
 * Parse the node config file.
 * 
 */
void
NodeManager::parseNodeConfig(const char *hostFile) {
    std::ifstream inFile(hostFile); 
    std::string ip, name, role;
    while (inFile >> ip >> name >> role) {
        if (ip == me.ip) {
            me.id = allNodes.size();
            me.name = name;
            me.master = (role == MASTER_ROLE);
        }
        masterId = (role == MASTER_ROLE) ? allNodes.size() : masterId;
        allNodes.push_back(Node(allNodes.size(), &ip, &name, (role == MASTER_ROLE)));
    }

    numLiveNodes = allNodes.size();
}


/**
 *
 * Handler when something happens to a node specified by given path.
 *
 */
void
NodeManager::nodeManagerCB(const char *path) {
    lockPhase.lock();
    switch (phase) {
        case REGISTERING:
            countChildren(path);
            break;
        case REGISTERED:
            printLog(me.id, "Something happened on node %s. Current phase is REGISTERED, hence do nothing...\n", getNodeName(path).c_str());
            break;
        case PROCESSING:
            nodeUpDown(path);
            break;
        case UNREGISTERING:
            printLog(me.id, "Something happened on node %s. Current phase in UNREGISTERING, anyways death is upon us now, hence do nothing...\n", getNodeName(path).c_str());
            break;
        default:
            assert(false);
    }
    lockPhase.unlock();
}


/**
 *
 * Sentence the node specified by given path to death.
 * 
 */
void
NodeManager::nodeUpDown(const char *path) {
    std::string nodeName = getNodeName(path);
    unsigned nId = getNodeId(nodeName);

    assert(nId < allNodes.size());

    std::string subNode = ZK_MASTER_NODE;
    subNode += "/";
    subNode += allNodes[nId].name;

    assert(!ZKInterface::checkZKExists(subNode.c_str(), nodeManagerCB));    // Ensure it is not a false alarm.
    assert(allNodes[nId].isAlive);

    // Send it to death, and if it was the master node, assign another living node as new master.
    printLog(me.id, "Node %u is sentenced to death...\n", nId);
    allNodes[nId].isAlive = false;
    if (allNodes[nId].master) {
        while (1) {
            unsigned nextId = (nId + 1) % allNodes.size();
            if (allNodes[nextId].isAlive) {
                allNodes[nextId].master = true;
                allNodes[nId].master = false;
                if (me.id == nextId) {
                    me.master = true;
                    printLog(me.id, "New master node is now me (%u).\n", me.id);
                }
                break;
            }
        }
    }
    --numLiveNodes;
    ndFunc(getNodeId(nodeName));
}


/**
 *
 * Watch all nodes and ensure they are alive in the ZooKeeprer.
 * 
 */
void
NodeManager::watchAllNodes() {
    for (unsigned i = 0; i < allNodes.size(); ++i) {
        std::string subNode = ZK_MASTER_NODE;
        subNode += "/";
        subNode += allNodes[i].name;

        assert(ZKInterface::checkZKExists(subNode.c_str(), nodeManagerCB));

        allNodes[i].isAlive = true;
    }
}


/**
 *
 * Count the registered children and wake them up if all have been settled.
 * 
 */
void
NodeManager::countChildren(const char *path) {
    struct String_vector children;
    ZKInterface::getZKNodeChildren(ZK_MASTER_NODE, nodeManagerCB, &children);

    // Watch all my children and see how many of them are registered. If all have been registered then
    // wake the waiting nodes up.
    if ((unsigned) children.count == allNodes.size()) {
        watchAllNodes();

        printLog(me.id, "Everyone (%d nodes) have been registered.\n", children.count);

        phase = REGISTERED;

        // Hack: Remove child watch on ZK_MASTER_NODE by creating and destroying a child.
        std::string subNode = ZK_MASTER_NODE "/" ZK_JUNK_NODE_NAME;
        createNode(subNode.c_str(), false, true, &createCB);
        ZKInterface::deleteZKNode(subNode.c_str());

        lockWaiter.lock();
        condWaiter.signal();
        lockWaiter.unlock();
    } else {
        assert((unsigned) children.count < allNodes.size());
        printLog(me.id, "A subset (%d nodes) have been registered.\n", children.count);
    }

    ZKInterface::freeZKStringVector(&children);
}


/**
 *
 * Wait for all nodes to get registered.
 * 
 */
void
NodeManager::waitForAllNodes() {
    struct String_vector children;
    ZKInterface::getZKNodeChildren(ZK_MASTER_NODE, nodeManagerCB, &children);

    // If not all nodes have been registered, block until then.
    if ((unsigned) children.count != allNodes.size()) {
        lockWaiter.lock();
        condWaiter.wait();
        lockWaiter.unlock();
    } else {
        watchAllNodes();

        printLog(me.id, "Everyone (%d nodes) have been registered (head start).\n", children.count);
        
        phase = REGISTERED;

        // Hack: Remove child watch on ZK_MASTER_NODE by creating and destroying a child.
        std::string subNode = ZK_MASTER_NODE "/" ZK_JUNK_NODE_NAME;
        createNode(subNode.c_str(), false, true, &createCB);
        ZKInterface::deleteZKNode(subNode.c_str());
    }

    ZKInterface::freeZKStringVector(&children);
}


/**
 *
 * Hit / leave a global barrier (for private use).
 * 
 */
void
NodeManager::checkNode(const char *path) {      // Checker function for ZooKeeper checkZKExists().
    lockWaiter.lock();
    condWaiter.signal();
    lockWaiter.unlock();
}

void
NodeManager::hitBarrier() {
    printLog(me.id, "Barrier %s: Hitting on it...\n", ZK_BARRIER_NODE);
    lockWaiter.lock();
    if (!ZKInterface::checkZKExists(ZK_BARRIER_NODE, checkNode))
        condWaiter.wait();
    lockWaiter.unlock();
    printLog(me.id, "Barrier %s: Entered.\n", ZK_BARRIER_NODE);
}

void
NodeManager::leaveBarrier() {
    printLog(me.id, "Barrier %s: Waiting to leave...\n", ZK_BARRIER_NODE);
    lockWaiter.lock();
    if (ZKInterface::checkZKExists(ZK_BARRIER_NODE, checkNode))
        condWaiter.wait();
    lockWaiter.unlock();
    printLog(me.id, "Barrier %s: Passed through!\n", ZK_BARRIER_NODE);
}


/**
 *
 * Create a new ZK node on the given path.
 * 
 */
void
NodeManager::createCB(int rc, const char *createdPath, const void *data) {  // Checker function for ZooKeeper createNode().
    assert(rc == 0);
}

void
NodeManager::createNode(const char *path, bool ephemeral, bool sync, void (*func)(int, const char *, const void *)) {
    ZKInterface::createZKNode(path, ephemeral, sync, func);
} 


/**
 *
 * Change a node's phase to processing.
 * 
 */
void
NodeManager::startProcessing() {
    lockPhase.lock();
    phase = PROCESSING;
    printLog(me.id, "Start processing. (Current phase -> PROCESSING)\n");
    lockPhase.unlock();
}


/**
 *
 * Handler when a barrier appears in the way.
 * 
 */
void
NodeManager::checkBarrier(const char *path) {   // Checker function for ZooKeeper checkZKExists().
    lockWaiter.lock();
    condWaiter.signal();
    lockWaiter.unlock();
}

void
NodeManager::barrierCB(const char *path) {
    lockAppBarriers.lock();   // Serializing the whole CB because anyways its a barrier, do not care about performance.
    
    assert(appBarriers.find(path) != appBarriers.end());
    
    if (appBarriers[path].ignoreCB) {   // I am set to ignore this handler.
        lockAppBarriers.unlock(); 
        return;
    }

    struct String_vector children;
    ZKInterface::getZKNodeChildren(path, barrierCB, &children);

    // If all have reached the barrier, remove it, and ignore it in the future.
    if ((unsigned) children.count == numLiveNodes - 1) {
        printLog(me.id, "Everyone reached the barrier, thus removing it...\n", children.count);
        appBarriers[path].ignoreCB = true;
        lockWaiter.lock();
        condWaiter.signal();
        lockWaiter.unlock();
    }

    ZKInterface::freeZKStringVector(&children);
    lockAppBarriers.unlock();
}
