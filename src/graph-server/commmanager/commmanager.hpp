#ifndef __COMM_MANAGER_HPP__
#define __COMM_MANAGER_HPP__


#include <set>
#include <vector>
#include <climits>
#include <zmq.hpp>
#include "../parallel/lock.hpp"
#include "../utils/utils.hpp"


#define NULL_CHAR MAX_IDTYPE


/** Control message topic & contents. */
#define CONTROL_MESSAGE_TOPIC 'C'
enum ControlMessageType { CTRLNONE = -1, IAMUP = -2, ISEEYOUUP = -3, APPMSG = -4 };


/** Structure of a controlled message. */
typedef struct controlMessage {
    char topic;
    ControlMessageType messageType;
    controlMessage(ControlMessageType mType = CTRLNONE) : topic(CONTROL_MESSAGE_TOPIC), messageType(mType) { }
} ControlMessage;


/**
 *
 * Class of the communication manager. Responsible for communications between nodes.
 * 
 */
class CommManager {

public:

    static void init();
    static void destroy();

    static void dataPushOut(unsigned topic, void* value, unsigned valSize);
    static bool dataPullIn(unsigned *topic, void *value, unsigned maxValSize);
    static void controlPushOut(unsigned to, void* value, unsigned valSize); 
    static bool controlPullIn(unsigned from, void *value, unsigned maxValSize);

    static void setDataPort(unsigned dPort) { dataPort = dPort; }
    static void setControlPortStart(unsigned cPort) { controlPortStart = cPort; }

private:

    static unsigned numNodes;
    static unsigned nodeId;


    static zmq::context_t dataContext;
    static zmq::socket_t *dataPublisher;
    static zmq::socket_t *dataSubscriber;
    static unsigned dataPort;

    static Lock lockDataPublisher;
    static Lock lockDataSubscriber;

    static zmq::context_t controlContext;
    static zmq::socket_t **controlPublishers;
    static zmq::socket_t **controlSubscribers;
    static unsigned controlPortStart;

    static Lock *lockControlPublishers;
    static Lock *lockControlSubscribers;

    static void flushControl();
    static void flushData();
};


#endif //__COMM_MANAGER_HPP__
