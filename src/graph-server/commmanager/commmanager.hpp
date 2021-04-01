#ifndef __COMM_MANAGER_HPP__
#define __COMM_MANAGER_HPP__


#include <set>
#include <vector>
#include <climits>
#include <zmq.hpp>
#include "../parallel/lock.hpp"
#include "../utils/utils.hpp"
#include "../nodemanager/nodemanager.hpp"


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
    void init(NodeManager& nodeManager, unsigned ctxThds = 2);
    void destroy();

    void rawMsgPushOut(zmq::message_t &msg);
    void dataPushOut(unsigned receiver, unsigned sender, unsigned topic, void* value, unsigned valSize);
    bool dataPullIn(unsigned *sender, unsigned *topic, void *value, unsigned maxValSize);
    void controlPushOut(unsigned to, void* value, unsigned valSize);
    bool controlPullIn(unsigned from, void *value, unsigned maxValSize);

    void setDataPort(unsigned dPort) { dataPort = dPort; }
    void setControlPortStart(unsigned cPort) { controlPortStart = cPort; }

private:

    unsigned numNodes = 0;
    unsigned nodeId = 0;

    zmq::context_t dataContext;
    zmq::socket_t *dataPublisher = NULL;
    zmq::socket_t *dataSubscriber = NULL;
    unsigned dataPort;

    Lock lockDataPublisher;
    Lock lockDataSubscriber;

    zmq::context_t controlContext;
    zmq::socket_t **controlPublishers = NULL;
    zmq::socket_t **controlSubscribers = NULL;
    unsigned controlPortStart;

    Lock *lockControlPublishers = NULL;
    Lock *lockControlSubscribers = NULL;

    void flushControl();
    void flushData();
};


#endif //__COMM_MANAGER_HPP__
