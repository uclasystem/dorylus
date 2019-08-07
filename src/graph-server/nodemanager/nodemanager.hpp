#ifndef __NODE_MANAGER_HPP__
#define __NODE_MANAGER_HPP__


#include <string>
#include <vector>
#include <map>
#include <zmq.hpp>


// Master node is by default the first machien on dshMashinesFile list.
#define MASTER_NODEID 0


/** Node message topic & contents. */
#define NODE_MESSAGE_TOPIC 'N'
enum NodeMessageType { NODENONE = -1, MASTERUP = -2, WORKERUP = -3, INITDONE = -4, BARRIER = -5 };


/** Structure of a node managing message. */
typedef struct nodeMessage {
    char topic;
    NodeMessageType messageType;
    nodeMessage(NodeMessageType mType = NODENONE) : topic(NODE_MESSAGE_TOPIC), messageType(mType) { }
} NodeMessage;


/** Structure of a node information block. */
typedef struct node {
    unsigned id;
    std::string ip;
    std::string pubip;
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

    static void init(std::string dshMachinesFile, std::string myPrIpFile, std::string myPubIpFile);
    static void destroy();

    static void barrier();
    
    static Node& getNode(unsigned i);
    static unsigned getNumNodes();
    static unsigned getMyNodeId();
    static bool amIMaster();

    static void setNodePort(unsigned nPort) { nodePort = nPort; }

private:

    static Node me;
    static unsigned masterId;
    
    static std::vector<Node> allNodes;

    static bool inBarrier;

    static zmq::context_t nodeContext;
    static zmq::socket_t *nodePublisher;
    static zmq::socket_t *nodeSubscriber;
    static unsigned nodePort;

    static void parseNodeConfig(const std::string dshMachinesFile);
};


#endif //__NODE_MANAGER_HPP__
