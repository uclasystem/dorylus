#ifndef __NODE_MANAGER_HPP__
#define __NODE_MANAGER_HPP__


#include <string>
#include <vector>
#include <map>
#include <zmq.hpp>


class Engine;

// Master node is by default the first machien on dshMashinesFile list.
#define MASTER_NODEID 0


/** Node message topic & contents. */
#define NODE_MESSAGE_TOPIC 'N'
enum NodeMessageType { NODENONE = -1, MASTERUP = -2, WORKERUP = -3, INITDONE = -4, BARRIER = -5,
                       MINEPOCH = -6, MAXEPOCH = -7 };


/** Structure of a node managing message. */
typedef struct nodeMessage {
    char topic;
    NodeMessageType messageType;
    unsigned info;
    unsigned id;
    nodeMessage(NodeMessageType mType, unsigned _info = 0, unsigned _id = UINT_MAX)
      : topic(NODE_MESSAGE_TOPIC), messageType(mType), info(_info), id(_id) { }
} NodeMessage;


/** Structure of a node information block. */
typedef struct node {
    unsigned id;
    int localId;
    std::string ip;
    std::string prip;
    bool master;

    node() { }
    node(unsigned i, int lId, std::string *ip_, bool mtr) {
        id = i;
        localId = lId;
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
    void init(std::string workersFile, std::string myPrIpFile, Engine* _engine);
    void destroy();

    // Synchronous barrier
    void barrier();

    // Bounded staleness methods
    bool parseNodeMsg(NodeMessage &nMsg);
    void sendEpochUpdate(unsigned epoch);
    void readEpochUpdates();
    unsigned syncCurrEpoch(unsigned epoch);

    bool standAloneMode();
    Node& getNode(unsigned i);
    unsigned getNumNodes();
    unsigned getMyNodeId();
    bool amIMaster();

    void setNodePort(unsigned nPort) { nodePort = nPort; }

private:
    Node me;
    Engine* engine;
    unsigned masterId;
    unsigned numNodes;
    bool standAlone;


    std::vector<Node> allNodes;

    unsigned remaining;
    bool inBarrier = false;

    zmq::context_t nodeContext;
    zmq::socket_t *nodePublisher = NULL;
    zmq::socket_t *nodeSubscriber = NULL;
    unsigned nodePort;

    void parseNodeConfig(const std::string workersFile, int localId);
};


#endif //__NODE_MANAGER_HPP__
