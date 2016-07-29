#ifndef __node_manager_H__
#define __node_manager_H__

#include <string>
#include <vector>
#include <map>
#include <atomic>
#include <pthread.h>

#define MASTER_ROLE "master"
#define WORKER_ROLE "worker"

#define ZK_ROOT_NODE "/fancyapp"
#define ZK_MASTER_NODE ZK_ROOT_NODE"/nodemanager"
#define ZK_BARRIER_NODE ZK_ROOT_NODE"/barrier"

#define ZK_APPBARRIER_NODE  ZK_ROOT_NODE"/appbarriers"

#define NM_BARRIER_NODE "nmbarrier"

#define ZK_JUNK_NODE_NAME "zk_junk_node_name"

// REGISTERED phase is where registration has completed but processing hasn't started. 
// Here, we can do cleanup of REGISTERING phase like unwatching things.
enum Phase {REGISTERING, REGISTERED, PROCESSING, UNREGISTERING};


typedef struct node {
    unsigned id;
    std::string ip;
    std::string name;
    bool master;
    bool isAlive;

    node() { }
    node(unsigned i, std::string* ipx, std::string* n, bool mtr) {
        id = i;
        ip = *ipx;
        name = *n;
        master = mtr;
        isAlive = false;
    }
} Node;


typedef struct barrierContext {
    std::string path;
    bool ignoreCB;

    barrierContext() { }
    barrierContext(const char* p) : ignoreCB(false) { 
        path = ZK_APPBARRIER_NODE;
        path += "/";
        path += p;
    }
} BarrierContext;


class NodeManager {
    public:
    static bool init(const char* zooHostPort, const char* hostFile);
    static void registerNodeDownFunc(void (*func)(unsigned));

    static void destroy();
    static std::vector<Node>* getAllNodes();
    static Node getNode(unsigned i);
    static unsigned getNumNodes();
    static unsigned getNodeId();
    static void startProcessing();
    static void barrier(const char* bar);
    static void releaseAll(const char* bar);
    static bool amIMaster();

    private:
    static Node me;
    //static bool master;
    
    static std::vector<Node> allNodes;
    static unsigned numLiveNodes;

    static pthread_mutex_t mtx_waiter;
    static pthread_cond_t cond_waiter;

    static Phase phase;
    static pthread_mutex_t mtx_phase;

    static std::map<std::string, BarrierContext> appBarriers;
    static pthread_mutex_t mtx_appBarriers;

    static std::atomic<bool> forceRelease;
    static std::atomic<bool> inBarrier;

    static std::string zooHostPort;

    static void (*ndFunc)(unsigned);

    static void parseZooConfig(const char* zooHostFile);
    static void parseNodeConfig(const char* hostFile);

    static void nodeManagerCB(const char* path);    // This should be the main CB when nodes come and go
    static void nodeUpDown(const char* path);
    static void countChildren(const char* path);
    static void waitForAllNodes();
    static void watchAllNodes();
    static void checkExists(const char* path);
    static void hitBarrier();
    static void checkNotExists(const char* path);
    static void leaveBarrier();
    static void createNode(const char* path, bool ephemeral, bool sync, void (*func)(int, const char*, const void*)); 
    static void createCB(int rc, const char* createdPath, const void* data);
    //static void createKCB(int rc, const char* createdPath, const void* data);

    static void barrierCB(const char* path);
    static void checkBarrierExists(const char* path);
    static void checkBarrierNotExists(const char* path);
    static std::string getNodeName(const char* path);
    static unsigned getNodeId(std::string& nodeName);
};

#endif //__node_manager_H__
