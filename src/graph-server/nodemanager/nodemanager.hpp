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

    void init(std::string dshMachinesFile, std::string myPrIpFile, std::string myPubIpFile);
    void destroy();

    void barrier();
    
    bool standAloneMode();
    Node& getNode(unsigned i);
    unsigned getNumNodes();
    unsigned getMyNodeId();
    bool amIMaster();

    void setNodePort(unsigned nPort) { nodePort = nPort; }

private:

    Node me;
    unsigned masterId;
    bool standAlone;

    std::vector<Node> allNodes;

    bool inBarrier = false;

    zmq::context_t nodeContext;
    zmq::socket_t *nodePublisher = NULL;
    zmq::socket_t *nodeSubscriber = NULL;
    unsigned nodePort;
    
    void parseNodeConfig(const std::string dshMachinesFile);
};


#endif //__NODE_MANAGER_HPP__
