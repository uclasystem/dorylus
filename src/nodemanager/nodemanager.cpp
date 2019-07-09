#include "nodemanager.h"
#include "zkinterface.h"
#include "../utils/utils.h"
#include <fstream>
#include <cassert>
#include <cstring>

Node NodeManager::me;
unsigned NodeManager::masterIdx = 0;
std::vector<Node> NodeManager::allNodes;
unsigned NodeManager::numLiveNodes;
pthread_mutex_t NodeManager::mtx_waiter;
pthread_cond_t NodeManager::cond_waiter;
Phase NodeManager::phase = REGISTERING;
pthread_mutex_t NodeManager::mtx_phase; 
std::map<std::string, BarrierContext> NodeManager::appBarriers;
pthread_mutex_t NodeManager::mtx_appBarriers;
void (*NodeManager::ndFunc)(unsigned) = NULL;
std::string NodeManager::zooHostPort;
std::atomic<bool> NodeManager::forceRelease = ATOMIC_VAR_INIT(false);
std::atomic<bool> NodeManager::inBarrier = ATOMIC_VAR_INIT(false);

void NodeManager::parseZooConfig(const char* zooHostFile) {
    zooHostPort = "";
    bool first = true;
    std::ifstream inFile(zooHostFile);
    std::string host, port;
    while (inFile >> host >> port) {
        if(first == false)
            zooHostPort += ",";
        zooHostPort += host + ":" + port;
        first = false;
    }
}


void NodeManager::parseNodeConfig(const char* hostFile) {
    std::ifstream inFile(hostFile); 
    std::string ip, name, role;
    while (inFile >> ip >> name >> role) {
        if(ip == me.ip) {
            me.id = allNodes.size();
            me.name = name;
            me.master = (role == MASTER_ROLE);
        }
        masterIdx = (role == MASTER_ROLE) ? allNodes.size() : masterIdx;
        allNodes.push_back(Node(allNodes.size(), &ip, &name, (role == MASTER_ROLE)));
    }

    numLiveNodes = allNodes.size();
    for(unsigned i=0; i<allNodes.size(); ++i)
        fprintf(stderr, "%u %s %s\n", allNodes[i].id, allNodes[i].ip.c_str(), allNodes[i].name.c_str());
    fprintf(stderr, "Node %u is master\n", masterIdx);
}

void NodeManager::nodeManagerCB(const char* path) {
    pthread_mutex_lock(&mtx_phase);
    switch(phase) {
        case REGISTERING:
            countChildren(path);
            break;
        case REGISTERED:
            fprintf(stderr, "Something happened with node %s. Current phase is REGISTERED, hence do nothing.\n", getNodeName(path).c_str());
            //assert(false);
            break;
        case PROCESSING:
            nodeUpDown(path);
            break;
        case UNREGISTERING:
            fprintf(stderr, "Something happened with node %s. Current phase in UNREGISTERING. Anyways death is upon us now, hence do nothing.\n", getNodeName(path).c_str());
            break;
        default:
            assert(false);
    }
    pthread_mutex_unlock(&mtx_phase);
}

void NodeManager::nodeUpDown(const char* path) {
    std::string nodeName = getNodeName(path);
    unsigned nId = getNodeId(nodeName);
    fprintf(stderr, "Something happened to node %u\n", nId);
    if(nId < allNodes.size()) {
        std::string subNode = ZK_MASTER_NODE;
        subNode += "/";
        subNode += allNodes[nId].name;

        if(ZKInterface::checkZKExists(subNode.c_str(), nodeManagerCB) == false) {
            fprintf(stderr, "Node %s is down. calling ndFunc(%u)\n", nodeName.c_str(), getNodeId(nodeName));
            assert(allNodes[nId].isAlive);
            allNodes[nId].isAlive = false;
            if(allNodes[nId].master) {
                while(1) {
                    unsigned nextId = (nId + 1) % allNodes.size();
                    if(allNodes[nextId].isAlive) {
                        allNodes[nextId].master = true;
                        allNodes[nId].master = false;
                        if(me.id == nextId) {
                            me.master = true;
                            fprintf(stderr, "New master in town with id = %u\n", me.id);
                        }
                        break;
                    }
                }
            }
            --numLiveNodes;
            ndFunc(getNodeId(nodeName));
        }
        else {
            fprintf(stderr, "Some kind of false alarm for path %s (or %s)\n", path, subNode.c_str());
            assert(false);
        }
    } else
        assert(false);
}


void NodeManager::registerNodeDownFunc(void (*func)(unsigned)) {
    ndFunc = func;
}

void NodeManager::watchAllNodes() {
    for(unsigned i=0; i<allNodes.size(); ++i) {
        std::string subNode = ZK_MASTER_NODE;
        subNode += "/";
        subNode += allNodes[i].name;

        if(ZKInterface::checkZKExists(subNode.c_str(), nodeManagerCB)) {
            //fprintf(stderr, "%s exists\n", subNode.c_str());
            allNodes[i].isAlive = true;
        }
        else {
            //fprintf(stderr, "%s doesn't exist!\n", subNode.c_str());
            assert(false);
        } 
        //assert(ZKInterface::checkZKExists(subNode.c_str(), nodeManagerCB));
    } 
}

void NodeManager::countChildren(const char* path) {
    struct String_vector children;
    ZKInterface::getZKNodeChildren(ZK_MASTER_NODE, nodeManagerCB, &children);
    if((unsigned) children.count == allNodes.size()) {
        watchAllNodes();

        fprintf(stderr, "Everyone (%d nodes) registered.\n", children.count);
        phase = REGISTERED;

        // Hack: Remove child watch on ZK_MASTER_NODE by creating and destroying a child
        std::string subNode = ZK_MASTER_NODE "/" ZK_JUNK_NODE_NAME;
        createNode(subNode.c_str(), false, true, &createCB);
        ZKInterface::deleteZKNode(subNode.c_str());

        pthread_mutex_lock(&mtx_waiter);
        pthread_cond_signal(&cond_waiter);
        pthread_mutex_unlock(&mtx_waiter);
    } else {
        fprintf(stderr, "%d nodes registered.\n", children.count);
        assert((unsigned) children.count < allNodes.size());
    }

    ZKInterface::freeZKStringVector(&children);
}

void NodeManager::waitForAllNodes() {
    struct String_vector children;
    ZKInterface::getZKNodeChildren(ZK_MASTER_NODE, nodeManagerCB, &children);
    if((unsigned) children.count != allNodes.size()) {
        pthread_mutex_lock(&mtx_waiter);
        pthread_cond_wait(&cond_waiter, &mtx_waiter);
        pthread_mutex_unlock(&mtx_waiter);
    } else {
        watchAllNodes();
        fprintf(stderr, "Everyone (%d nodes) registered (this is a head start).\n", children.count);
        phase = REGISTERED;

        // Hack: Remove child watch on ZK_MASTER_NODE by creating and destroying a child
        std::string subNode = ZK_MASTER_NODE "/" ZK_JUNK_NODE_NAME;
        createNode(subNode.c_str(), false, true, &createCB);
        ZKInterface::deleteZKNode(subNode.c_str());
    }
    ZKInterface::freeZKStringVector(&children);
}


void NodeManager::checkExists(const char* path) {
    //fprintf(stderr, "Node %s exists now.\n", path);
    pthread_mutex_lock(&mtx_waiter);
    pthread_cond_signal(&cond_waiter);
    pthread_mutex_unlock(&mtx_waiter);
}

void NodeManager::hitBarrier() {
    fprintf(stderr, "Node %s: Waiting to enter barrier %s\n", me.name.c_str(), ZK_BARRIER_NODE);
    pthread_mutex_lock(&mtx_waiter);
    if(ZKInterface::checkZKExists(ZK_BARRIER_NODE, checkExists) == false)
        pthread_cond_wait(&cond_waiter, &mtx_waiter);
    pthread_mutex_unlock(&mtx_waiter);
    fprintf(stderr, "Node %s: Entered barrier %s\n", me.name.c_str(), ZK_BARRIER_NODE);
}

void NodeManager::checkNotExists(const char* path) {
    //fprintf(stderr, "Node %s doesn't exist now.\n", path);
    pthread_mutex_lock(&mtx_waiter);
    pthread_cond_signal(&cond_waiter);
    pthread_mutex_unlock(&mtx_waiter);
}

void NodeManager::leaveBarrier() {
    fprintf(stderr, "Node %s: Waiting to leave barrier %s\n", me.name.c_str(), ZK_BARRIER_NODE);
    pthread_mutex_lock(&mtx_waiter);
    if(ZKInterface::checkZKExists(ZK_BARRIER_NODE, checkNotExists) == true)
        pthread_cond_wait(&cond_waiter, &mtx_waiter);
    pthread_mutex_unlock(&mtx_waiter);
    fprintf(stderr, "Node %s: Left barrier %s\n", me.name.c_str(), ZK_BARRIER_NODE);
}

void NodeManager::createCB(int rc, const char* createdPath, const void* data) {
    if(rc != 0)
        fprintf(stderr, "rc != 0 for path %s. It is %d\n", createdPath, rc);
    assert(rc == 0);
    //fprintf(stderr, "Node %s created.\n", createdPath);
}

/*
void NodeManager::createKCB(int rc, const char* createdPath, const void* data) {
    if(rc != 0)
        fprintf(stderr, "rc != 0 for path %s. It is %d\n", createdPath, rc);
    assert(rc == 0);
    fprintf(stderr, "Node %s created.\n", createdPath);
}
*/

void NodeManager::createNode(const char* path, bool ephemeral, bool sync, void (*func)(int, const char*, const void*)) {
    //ZKInterface::createZKNode(path, ephemeral, sync, &createCB);
    ZKInterface::createZKNode(path, ephemeral, sync, func);
} 

void NodeManager::startProcessing() {
    pthread_mutex_lock(&mtx_phase);
    phase = PROCESSING;
    fprintf(stderr, "Current phase = PROCESSING\n");
    pthread_mutex_unlock(&mtx_phase);
}

void NodeManager::checkBarrierExists(const char* path) {
    //fprintf(stderr, "Node %s exists now.\n", path);
    pthread_mutex_lock(&mtx_waiter);
    pthread_cond_signal(&cond_waiter);
    pthread_mutex_unlock(&mtx_waiter);
}

void NodeManager::checkBarrierNotExists(const char* path) {
    //fprintf(stderr, "Node %s doesn't exist now.\n", path);
    pthread_mutex_lock(&mtx_waiter);
    pthread_cond_signal(&cond_waiter);
    pthread_mutex_unlock(&mtx_waiter);
}


void NodeManager::barrierCB(const char* path) {
    pthread_mutex_lock(&mtx_appBarriers);   // Serializing whole CB because anyways its a barrier -- no worry about performance
    assert(appBarriers.find(path) != appBarriers.end());
    if(appBarriers[path].ignoreCB) {
        //fprintf(stderr, "barrierCB ignoring\n");
        pthread_mutex_unlock(&mtx_appBarriers); 
        return;
    }
    struct String_vector children;
    //fprintf(stderr, "barrierCB checking for number of children\n");
    ZKInterface::getZKNodeChildren(path, barrierCB, &children);
    //if((unsigned) children.count == allNodes.size() - 1) {
    if((unsigned) children.count == numLiveNodes - 1) {
        fprintf(stderr, "Everyone (%d nodes apart from master) reached barrier\n", children.count);
        assert(appBarriers.find(path) != appBarriers.end());
        appBarriers[path].ignoreCB = true;
        pthread_mutex_lock(&mtx_waiter);
        pthread_cond_signal(&cond_waiter);
        pthread_mutex_unlock(&mtx_waiter);
    } else {
        fprintf(stderr, "%d nodes (apart from master) reached barrier\n", children.count);
    }
    ZKInterface::freeZKStringVector(&children);
    pthread_mutex_unlock(&mtx_appBarriers);
}

void NodeManager::barrier(const char* bar) {
    inBarrier = true;

    BarrierContext bContext(bar);
    std::string barName = bContext.path;
    pthread_mutex_lock(&mtx_appBarriers);
    appBarriers[barName] = bContext;    // Overwrite barrier
    pthread_mutex_unlock(&mtx_appBarriers);

    if(me.master) {
        createNode(barName.c_str(), false, true, &createCB);
        //fprintf(stderr, "Node %s: Created barrier %s\n", me.name.c_str(), barName.c_str());

        struct String_vector children;
        pthread_mutex_lock(&mtx_waiter);
        ZKInterface::getZKNodeChildren(barName.c_str(), barrierCB, &children);
        //if((unsigned) children.count != allNodes.size() - 1) {
        if((unsigned) children.count != numLiveNodes - 1) {
            pthread_cond_wait(&cond_waiter, &mtx_waiter);
            pthread_mutex_unlock(&mtx_waiter);
        } else {
            pthread_mutex_unlock(&mtx_waiter);
            fprintf(stderr, "Everyone (%d nodes apart from master) reached barrier (this is head start)\n", children.count);
            pthread_mutex_lock(&mtx_appBarriers);
            appBarriers[barName].ignoreCB = true; 
            pthread_mutex_unlock(&mtx_appBarriers);
        }
        ZKInterface::freeZKStringVector(&children);

        /*
           pthread_mutex_lock(&mtx_appBarriers);
           assert(appBarriers[barName].ignoreCB == true);
           pthread_mutex_unlock(&mtx_appBarriers);
         */
        ZKInterface::recursiveDeleteZKNode(barName.c_str());
    } else {
        fprintf(stderr, "Node %s: Waiting to enter barrier %s\n", me.name.c_str(), barName.c_str());
        pthread_mutex_lock(&mtx_waiter);
        if(ZKInterface::checkZKExists(barName.c_str(), checkBarrierExists) == false)
            pthread_cond_wait(&cond_waiter, &mtx_waiter);
        pthread_mutex_unlock(&mtx_waiter);
        fprintf(stderr, "Node %s: Entered barrier %s\n", me.name.c_str(), barName.c_str());

        if(forceRelease == false) {
        std::string subNode = barName;
        subNode += "/";
        subNode += me.name;
        createNode(subNode.c_str(), true, false, &createCB);
        }

        fprintf(stderr, "Node %s: Waiting to leave barrier %s\n", me.name.c_str(), barName.c_str());
        if(forceRelease == false) {
        pthread_mutex_lock(&mtx_waiter);
        if(ZKInterface::checkZKExists(barName.c_str(), checkBarrierNotExists) == true)
            pthread_cond_wait(&cond_waiter, &mtx_waiter);
        pthread_mutex_unlock(&mtx_waiter);
        }
        fprintf(stderr, "Node %s: Left barrier %s\n", me.name.c_str(), barName.c_str());

        if(me.master)   // This is possible because master itself died and I got re-elected as new master. 
            ZKInterface::recursiveDeleteZKNode(barName.c_str());
    }

    inBarrier = false;
    forceRelease = false;
    /*
       pthread_mutex_lock(&mtx_appBarriers);
       appBarriers.erase(barName);
       pthread_mutex_unlock(&mtx_appBarriers);
     */
    }

    void NodeManager::releaseAll(const char* bar) {
        assert(phase == PROCESSING);

        BarrierContext bContext(bar);
        std::string barName = bContext.path; 
        pthread_mutex_lock(&mtx_appBarriers);
        if(appBarriers.find(barName.c_str()) != appBarriers.end())
            appBarriers[barName.c_str()].ignoreCB = true;
        pthread_mutex_unlock(&mtx_appBarriers);

        pthread_mutex_lock(&mtx_waiter);
        forceRelease = true;
        pthread_cond_signal(&cond_waiter);
        pthread_mutex_unlock(&mtx_waiter);

        forceRelease = inBarrier ? true : false;
    }

    /*
Master:
0. Create ZK_MASTER_NODE 
1. Create ZK_BARRIER_NODE
2. Create ZK_MASTER_NODE/<name>
3. Wait for nodes to be created
4. Delete ZK_BARRIER_NODE
Worker:
1. Wait for ZK_BARRIER_NODE to be created
2. Create ZK_MASTER_NODE/<name>
3. Wait for ZK_BARRIER_NODE to be destoyed
     */
    bool NodeManager::init(const char* zooHostFile, const char* hostFile) {
        getIP(&me.ip);
        parseZooConfig(zooHostFile);
        parseNodeConfig(hostFile);

        fprintf(stderr, "NodeManager initing zk with: %s\n", zooHostPort.c_str());
        assert(ZKInterface::init(zooHostPort.c_str()));

        pthread_mutex_init(&mtx_waiter, NULL);
        pthread_cond_init(&cond_waiter, NULL);

        pthread_mutex_init(&mtx_phase, NULL);
        pthread_mutex_init(&mtx_appBarriers, NULL);

        if(me.master) {
            ZKInterface::recursiveDeleteZKNode(ZK_ROOT_NODE);
            createNode(ZK_ROOT_NODE, false, true, &createCB);
            createNode(ZK_MASTER_NODE, false, true, &createCB);  // ephemeral? sync?
            createNode(ZK_APPBARRIER_NODE, false, false, &createCB);
            createNode(ZK_BARRIER_NODE, false, false, &createCB);

            std::string subNode = ZK_MASTER_NODE;
            subNode += "/";
            subNode += me.name;
            createNode(subNode.c_str(), true, false, &createCB);

            waitForAllNodes();  // block till everyone gets registered

            ZKInterface::deleteZKNode(ZK_BARRIER_NODE);
        } else {
            hitBarrier();   // block till the barrier appears

            std::string subNode = ZK_MASTER_NODE;
            subNode += "/";
            subNode += me.name;
            ZKInterface::createZKNode(subNode.c_str(), true, false, &createCB);

            leaveBarrier(); // block till the barrier disappears
            watchAllNodes();    // Is this required? Basically whenever a node goes down, everyone comes to know?
            phase = REGISTERED;
        }
        return true;
    }

    void NodeManager::destroy() {
        phase = UNREGISTERING;
        barrier("destroy"); 
        if(me.master) {
            ZKInterface::recursiveDeleteZKNode(ZK_ROOT_NODE);
        } 
    }

    std::vector<Node>* NodeManager::getAllNodes() {
        return &allNodes;
    }

    Node NodeManager::getNode(unsigned i) {
        return allNodes[i];
    }

    unsigned NodeManager::getNumNodes() {
        return allNodes.size();
    }

    unsigned NodeManager::getNodeId() {
        return me.id; 
    }

    bool NodeManager::amIMaster() {
        return me.master;
    }

    unsigned NodeManager::masterId() {
        return masterIdx; 
    }

    unsigned NodeManager::getNodeId(std::string& nodeName) {
        for(unsigned i=0; i<allNodes.size(); ++i)
            if(allNodes[i].name == nodeName)
                return i;

        //assert(false);
        return allNodes.size();
    }

    std::string NodeManager::getNodeName(const char* path) {
        char* cptr = const_cast<char*>(path) + strlen(path) - 1;
        assert(*cptr != '/');
        while(cptr != path) {
            --cptr;
            if(*cptr == '/') {
                ++cptr;
                break;
            }
        }
        fprintf(stderr, "getNodeName of %s is %s\n", path, cptr);
        return std::string(cptr);
    }
