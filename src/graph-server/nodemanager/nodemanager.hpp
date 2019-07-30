#ifndef __NODE_MANAGER_HPP__
#define __NODE_MANAGER_HPP__


#include <string>
#include <vector>
#include <map>
#include <atomic>
#include <pthread.h>
#include "../parallel/lock.hpp"
#include "../parallel/cond.hpp"


#define MASTER_ROLE "master"
#define WORKER_ROLE "worker"
#define ZK_JUNK_NODE_NAME "zk_junk_node_name"


/** Barrier path location. */
#define ZK_ROOT_NODE "/fancyapp"
#define ZK_MASTER_NODE      ZK_ROOT_NODE"/nodemanager"
#define ZK_BARRIER_NODE     ZK_ROOT_NODE"/barrier"
#define ZK_APPBARRIER_NODE  ZK_ROOT_NODE"/appbarriers"


#define DESTROY_BARRIER "destroy" 


/** Execution phase of a node. */
// REGISTERED phase is where registration has completed but processing hasn't started. 
// Here, we can do clean-up of REGISTERING phase like unwatching things.
enum Phase { REGISTERING, REGISTERED, PROCESSING, UNREGISTERING };


/** Structure of a node information block. */
typedef struct node {
    unsigned id;
    std::string ip;
    std::string pubip;
    std::string name;
    bool master;
    bool isAlive;

    node() { }
    node(unsigned i, std::string *ipx, std::string *pip, std::string *n, bool mtr) {
        id = i;
        ip = *ipx;
        pubip = *pip;
        name = *n;
        master = mtr;
        isAlive = false;
    }
} Node;


/** Structure wrapping over a barrier's path. */
typedef struct barrierContext {
    std::string path;
    bool ignoreCB;

    barrierContext() { }
    barrierContext(const char *p) : ignoreCB(false) { 
        path = ZK_APPBARRIER_NODE;
        path += "/";
        path += p;
    }
} BarrierContext;


/**
 *
 * Class of the node manager. Reponsible for handling my machine's role in the cluster.
 * 
 */
class NodeManager {

public:

    static void init(const char *zooHostPort, const char *hostFile);
    static void destroy();
    static std::vector<Node>* getAllNodes();
    static Node getNode(unsigned i);
    static unsigned getNumNodes();
    static unsigned getNodeId();
    static void startProcessing();
    static void barrier(const char *bar);
    static void releaseAll(const char *bar);
    static bool amIMaster();
    static unsigned getMasterId();
    static void registerNodeDownFunc(void (*func)(unsigned));

private:

    static Node me;
    static unsigned masterId;
    
    static std::vector<Node> allNodes;
    static unsigned numLiveNodes;

    static Lock lockWaiter;
    static Cond condWaiter;

    static Phase phase;
    static Lock lockPhase;

    static std::map<std::string, BarrierContext> appBarriers;
    static Lock lockAppBarriers;

    static std::atomic<bool> forceRelease;
    static std::atomic<bool> inBarrier;

    static std::string zooHostPort;

    static void (*ndFunc)(unsigned);

    static void parseZooConfig(const char *zooHostFile);
    static void parseNodeConfig(const char *hostFile);

    static void nodeManagerCB(const char *path);
    static void nodeUpDown(const char *path);

    static void watchAllNodes();
    static void countChildren(const char *path);
    static void waitForAllNodes();

    static void checkNode(const char *path);
    static void hitBarrier();
    static void leaveBarrier();

    static void createNode(const char *path, bool ephemeral, bool sync, void (*func)(int, const char *, const void *)); 
    static void createCB(int rc, const char *createdPath, const void *data);

    static void checkBarrier(const char *path);
    static void barrierCB(const char *path);
    
    static std::string getNodeName(const char *path);
    static unsigned getNodeId(std::string& nodeName);
};


#endif //__NODE_MANAGER_HPP__
