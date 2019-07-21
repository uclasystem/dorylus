#ifndef __COMM_MANAGER_H__
#define __COMM_MANAGER_H__


#include <set>
#include <vector>
#include <climits>
#include <zmq.hpp>
#include "../parallel/lock.hpp"
#include "../utils/utils.hpp"


#define LOCAL_HOST "*" //"localhost"
#define DATA_PORT 5000
#define DATA_PORT_STR "5000"  // This is where any data topic is allowed
//#define GENERIC_DATA_ADDR LOCAL_HOST ":" GENERIC_DATA_PORT

#define CONTROL_PORT_START 6000 // This is where P2P communication happens
#define CONTROL_PORT_START_STR "6000"

#define INF_WM 0

#define CONTROL_MESSAGE_TOPIC 'C'

#define SUBSCRIBE_BARRIER "subscribe"
#define SUBSCRIBEDONE_BARRIER "subscribedone"

#define NULL_CHAR MAX_IDTYPE
#define SUBSCRIPTION_COMPLETE MAX_IDTYPE - 1 

enum ControlMessageType {NONE = -1, IAMUP = -2, ISEEYOUUP = -3, SUBSCRIPTIONDONE = -4, APPMSG = -5};

typedef struct controlMessage {
    char topic;
    ControlMessageType messageType;
    controlMessage(ControlMessageType mType = NONE) : topic(CONTROL_MESSAGE_TOPIC), messageType(mType) { }
} ControlMessage;

class CommManager {
public:
    static void init();
    static void destroy();
    static void subscribeData(std::set<IdType>* topics, std::vector<IdType>* outTopics);
    static void dataPushOut(IdType topic, void* value, unsigned valSize);
    static bool dataPullIn(IdType& topic, std::vector<FeatType>& value);
    static void dataSyncPullIn(IdType& topic, std::vector<FeatType>& value);
    static void controlPushOut(unsigned to, void* value, unsigned valSize); 
    static bool controlPullIn(unsigned from, void* value, unsigned valSize);
    static void controlSyncPullIn(unsigned from, void* value, unsigned valSize);

    static void flushDataControl();
    static void flushData(unsigned nNodes);
    static void flushControl();
    static void flushControl(unsigned i);
    static void nodeDied(unsigned nId);

    static void setDataPort(unsigned dPort) { dataPort = dPort; fprintf(stderr, "dataPort set to %u\n", dataPort); }
    static void setControlPortStart(unsigned cPort) { controlPortStart = cPort; fprintf(stderr, "controlPortStart set to %u\n", controlPortStart); }

private:

    static unsigned numNodes;
    static unsigned nodeId;

    static unsigned dataPort;
    static unsigned controlPortStart;

    static zmq::context_t dataContext;
    static zmq::socket_t *dataPublisher;
    static zmq::socket_t *dataSubscriber;

    static Lock lockDataPublisher;
    static Lock lockDataSubscriber;

    static zmq::context_t controlContext;
    static zmq::socket_t **controlPublishers;
    static zmq::socket_t **controlSubscribers;

    static Lock *lockControlPublishers;
    static Lock *lockControlSubscribers;

    static std::vector<bool> nodesAlive;
    static unsigned numLiveNodes;
};

#endif //__COMM_MANAGER_H__
