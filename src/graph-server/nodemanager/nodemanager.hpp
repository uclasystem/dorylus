#ifndef __NODE_MANAGER_HPP__
#define __NODE_MANAGER_HPP__


#include <string>
#include <vector>
#include <map>


// Master node is by default the first machien on dshMashinesFile list.
#define MASTER_NODEID 0


/** Structure of a node information block. */
typedef struct node {
    unsigned id;
    std::string ip;
    bool master;

    node() { }
    node(unsigned i, std::string *ip_, bool mtr) {
        id = i;
        ip = *ip_;
        master = mtr;
    }
} Node;


/**
 *
 * Class of the node manager. Reponsible for handling my machine's role in the cluster.
 * 
 */
class NodeManager {

public:

    static void init(const std::string dshMachinesFile);
    static void destroy();
    static void barrier();
    static Node& getNode(unsigned i);
    static unsigned getNumNodes();
    static unsigned getNodeId();
    static bool amIMaster();
    static unsigned getMasterId();

private:

    static Node me;
    static unsigned masterId;
    
    static std::vector<Node> allNodes;

    static void parseNodeConfig(const char *hostFile);
};


#endif //__NODE_MANAGER_HPP__
