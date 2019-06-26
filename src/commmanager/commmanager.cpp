#include <vector>
#include <unistd.h>

#include <zmq.hpp>

#include "commmanager.h"
#include "../nodemanager/nodemanager.h"

unsigned CommManager::numNodes = 0;
unsigned CommManager::nodeId = 0;

// data sockets and mutex's
zmq::context_t CommManager::dataContext;
zmq::socket_t* CommManager::dataPublisher = NULL;
zmq::socket_t* CommManager::dataSubscriber = NULL;
pthread_mutex_t CommManager::mtx_dataPublisher;
pthread_mutex_t CommManager::mtx_dataSubscriber;

// control sockets and mutex's
zmq::socket_t** CommManager::controlPublishers = NULL;
zmq::socket_t** CommManager::controlSubscribers = NULL;
zmq::context_t CommManager::controlContext;
pthread_mutex_t* CommManager::mtx_controlPublishers = NULL;
pthread_mutex_t* CommManager::mtx_controlSubscribers = NULL;

std::vector<bool> CommManager::nodesAlive;
unsigned CommManager::numLiveNodes;
unsigned CommManager::dataPort = DATA_PORT;
unsigned CommManager::controlPortStart = CONTROL_PORT_START;


bool CommManager::init() {
    numNodes = NodeManager::getNumNodes();
    nodeId = NodeManager::getNodeId();

    Node me = NodeManager::getNode(nodeId);
    dataPublisher = new zmq::socket_t(dataContext, ZMQ_PUB);
    assert(dataPublisher->ksetsockopt(ZMQ_SNDHWM, INF_WM));
    assert(dataPublisher->ksetsockopt(ZMQ_RCVHWM, INF_WM));
    char hostPort[50];
    sprintf(hostPort, "tcp://%s:%u", me.ip.c_str(), dataPort);
    assert(dataPublisher->kbind(hostPort));
    fprintf(stderr, "Data bind complete\n");
    dataSubscriber = new zmq::socket_t(dataContext, ZMQ_SUB);
    assert(dataSubscriber->ksetsockopt(ZMQ_SNDHWM, INF_WM));
    assert(dataSubscriber->ksetsockopt(ZMQ_RCVHWM, INF_WM));

    for(unsigned i=0; i<numNodes; ++i) {
        /*
           if(i == nodeId)   // After failure, same node can have remote data
           continue;
         */

        // Connect to that node
        Node node = NodeManager::getNode(i);
        char hostPort[50];
        sprintf(hostPort, "tcp://%s:%u", node.ip.c_str(), dataPort);
        dataSubscriber->connect(hostPort);

        nodesAlive.push_back(true);
    }
    numLiveNodes = numNodes;

    /*IdType tpc = SUBSCRIPTION_COMPLETE;
    dataSubscriber->setsockopt(ZMQ_SUBSCRIBE, &tpc, sizeof(IdType));
    tpc = NULL_CHAR;
    dataSubscriber->setsockopt(ZMQ_SUBSCRIBE, &tpc, sizeof(IdType));
    */

    dataSubscriber->setsockopt(ZMQ_SUBSCRIBE, NULL, 0);

    controlPublishers = new zmq::socket_t*[numNodes];
    controlSubscribers = new zmq::socket_t*[numNodes];
    mtx_controlPublishers = new pthread_mutex_t[numNodes];
    mtx_controlSubscribers = new pthread_mutex_t[numNodes];

    for(unsigned i=0; i<numNodes; ++i) {
        if(i == nodeId)
            continue;

        controlPublishers[i] = new zmq::socket_t(controlContext, ZMQ_PUB);
        assert(controlPublishers[i]->ksetsockopt(ZMQ_SNDHWM, INF_WM));
        assert(controlPublishers[i]->ksetsockopt(ZMQ_RCVHWM, INF_WM));

        char hostPort[50];
        int prt = controlPortStart + i; 
        //sprintf(hostPort, "tcp://%s:%d", LOCAL_HOST, prt);
        sprintf(hostPort, "tcp://%s:%d", me.ip.c_str(), prt);
        fprintf(stderr, "Control publisher %u binding to %s\n", i, hostPort);
        assert(controlPublishers[i]->kbind(hostPort));

        controlSubscribers[i] = new zmq::socket_t(controlContext, ZMQ_SUB);
        assert(controlSubscribers[i]->ksetsockopt(ZMQ_SNDHWM, INF_WM));
        assert(controlSubscribers[i]->ksetsockopt(ZMQ_RCVHWM, INF_WM));

        Node node = NodeManager::getNode(i);
        sprintf(hostPort, "tcp://%s:%d", node.ip.c_str(), controlPortStart + me.id);
        fprintf(stderr, "Control subscriber %u connecting to %s\n", i, hostPort);
        controlSubscribers[i]->connect(hostPort);
        char tpc = CONTROL_MESSAGE_TOPIC;
        controlSubscribers[i]->setsockopt(ZMQ_SUBSCRIBE, &tpc, 1); 
        
        pthread_mutex_init(&mtx_controlPublishers[i], NULL);
        pthread_mutex_init(&mtx_controlSubscribers[i], NULL); 
    }

    fprintf(stderr, "Data/control bind/subscribe complete\n");

    /*
       0. bool[numnodes] saying which have finished. ctr = numnodes - 1. bool[nodeid] = true.
       1. Send imup to someone
       2. if receive imup from same, then 
       2.1 send icu to that node once
       2.2 receive icu from that node
       2.3 bool[i] = true. also reduce ctr
       3. if ctr = 0, break
       4. move to next node whose bool[i] is false
     */

    bool subscribed[numNodes];
    for(unsigned i=0; i<numNodes; ++i)
        subscribed[i] = false;

    subscribed[nodeId] = true;
    unsigned remaining = numNodes - 1;

    double lastSents[numNodes];
    for(unsigned i=0; i<numNodes; ++i)
        lastSents[i] = -getTimer() - 1000.0;

    unsigned i = 0;
    while(1) {
        if((i == nodeId) || (subscribed[i])) {
            i = (i + 1) % numNodes;
            continue;
        }

        //fprintf(stderr, "Trying with %s\n", NodeManager::getNode(i).name.c_str());

        if(lastSents[i] + getTimer() > 500) {
            zmq::message_t outMsg(sizeof(ControlMessage));
            ControlMessage cMsg(IAMUP);
            *((ControlMessage*) outMsg.data()) = cMsg;
            controlPublishers[i]->ksend(outMsg);
            lastSents[i] = -getTimer();
        }

        zmq::message_t inMsg;
        if(controlSubscribers[i]->krecv(&inMsg, ZMQ_DONTWAIT)) {
            //fprintf(stderr, "Received something from %s\n", NodeManager::getNode(i).name.c_str());
            ControlMessage cMsg = *((ControlMessage*) inMsg.data());
            if(cMsg.messageType == IAMUP) {
                //fprintf(stderr, "Received IAMUP, hence sending ISEEYOUUP to %s\n", NodeManager::getNode(i).name.c_str());
                {
                    zmq::message_t outAckMsg(sizeof(ControlMessage));
                    ControlMessage cAckMsg(ISEEYOUUP);
                    *((ControlMessage*) outAckMsg.data()) = cAckMsg;
                    controlPublishers[i]->ksend(outAckMsg);
                }

                while(1) {
                    zmq::message_t inAckMsg;
                    if(controlSubscribers[i]->krecv(&inAckMsg, ZMQ_DONTWAIT)) {
                        cMsg = *((ControlMessage*) inAckMsg.data());
                        //fprintf(stderr, "Received something from %s again\n", NodeManager::getNode(i).name.c_str());
                        if(cMsg.messageType == ISEEYOUUP) {
                            fprintf(stderr, "Control received ISEEYOUUP from %s\n", NodeManager::getNode(i).name.c_str());

                            zmq::message_t outAckMsg(sizeof(ControlMessage));
                            ControlMessage cAckMsg(ISEEYOUUP);
                            *((ControlMessage*) outAckMsg.data()) = cAckMsg;  
                            controlPublishers[i]->ksend(outAckMsg);

                            subscribed[i] = true;
                            --remaining;
                            fprintf(stderr, "Remaining = %u\n", remaining);
                            break;
                        }
                    } else 
                        break;
                }

            } else if (cMsg.messageType == ISEEYOUUP) {
                fprintf(stderr, "Control received ISEEYOUUP from %s (this is head-start)\n", NodeManager::getNode(i).name.c_str());

                zmq::message_t outAckMsg(sizeof(ControlMessage));
                ControlMessage cAckMsg(ISEEYOUUP);
                *((ControlMessage*) outAckMsg.data()) = cAckMsg;
                controlPublishers[i]->ksend(outAckMsg);

                subscribed[i] = true;
                --remaining;
                fprintf(stderr, "Remaining = %u\n", remaining);
            }
        }

        if(remaining == 0)
            break;

        i = (i + 1) % numNodes;
    }

    //pthread_mutex_init(&mtx_dataContext, NULL);
    pthread_mutex_init(&mtx_dataPublisher, NULL);
    pthread_mutex_init(&mtx_dataSubscriber, NULL); 
    //pthread_mutex_init(&mtx_controlContext, NULL);

    flushControl();
    return true;
}

void CommManager::destroy() {
    flushDataControl();

    dataPublisher->close();
    dataSubscriber->close();

    delete dataPublisher;
    delete dataSubscriber;

    //fprintf(stderr, "H0.0\n");

    for(unsigned i=0; i<numNodes; ++i) {
        if(i == nodeId)
            continue;

        controlPublishers[i]->close();
        controlSubscribers[i]->close();

        delete controlPublishers[i];
        delete controlSubscribers[i];
    
        pthread_mutex_destroy(&mtx_controlPublishers[i]);
        pthread_mutex_destroy(&mtx_controlSubscribers[i]);
    }

    //fprintf(stderr, "H0.1\n");

    delete[] controlPublishers;
    delete[] controlSubscribers;

    delete[] mtx_controlPublishers;
    delete[] mtx_controlSubscribers;

    //fprintf(stderr, "H0.2\n");

    dataContext.close();

    //fprintf(stderr, "H0.3\n");

    controlContext.close();

    //fprintf(stderr, "H0.4\n");
}

void CommManager::subscribeData(std::set<IdType>* topics, std::vector<IdType>* outTopics) {
    //pthread_mutex_lock(&mtx_dataContext);
    pthread_mutex_lock(&mtx_dataPublisher);
    pthread_mutex_lock(&mtx_dataSubscriber);

    zmq::message_t inMsg;

    /*std::set<IdType>::iterator it;
    for(it=topics->begin(); it != topics->end(); ++it) {
        IdType tpc = *it;
        dataSubscriber->setsockopt(ZMQ_SUBSCRIBE, (void*) &tpc, sizeof(IdType));
    }*/

    NodeManager::barrier(SUBSCRIBE_BARRIER);        // is this required? no?

    std::set<IdType> subscribed;

    unsigned rem = numLiveNodes - 1; bool firstTimeDone = true;
    while(rem > 0) {
        //fprintf(stderr, "Node %u here. outTopics->size() = %u\n", nodeId, outTopics->size());
        for(unsigned i=0; i<outTopics->size(); ++i) {
            zmq::message_t outMsg(sizeof(IdType));
            *((IdType*) outMsg.data()) = outTopics->at(i);
            dataPublisher->ksend(outMsg);
        }

        sleep(1);

        if(subscribed.size() != topics->size()) {
            //for(unsigned times = 0; times < numNodes * topics->size(); ++times) {
            while(1) {
                //zmq::message_t inMsg;
                if(dataSubscriber->krecv(&inMsg, ZMQ_DONTWAIT) == false) 
                    break;

                IdType idx = *((IdType*) inMsg.data());
                if(topics->find(idx) != topics->end()) {
                    //fprintf(stderr, "Hear topic %u\n", idx);
                    /*int ct = -1;
                      if(subscribed.find(idx) == subscribed.end())
                      ct = subscribed.size();
                     */
                    subscribed.insert(idx); 
                    /*
                       if(ct != -1) {
                    //if(nodeId != 0) fprintf(stderr, "Node %u : subscribed.size() = %zd and ct = %d", nodeId, subscribed.size(), ct);
                    assert(subscribed.size() == ct + 1);
                    }
                     */

                    if(subscribed.size() == topics->size())
                        break; // This break is so that we publish some more to allow others to listen
                } else {
                    // IGNORE!!
                    //fprintf(stderr, "Data subscription logic received: %u .. ignoring\n", idx);
                    //assert(false);
                }
            }
        }

        if(subscribed.size() == topics->size()) {
            if(firstTimeDone) {
                fprintf(stderr, "Node %u: Sending out SUBSCRIPTIONDONE\n", nodeId);
                for(unsigned i=0; i<numNodes; ++i) {
                    if((i == nodeId) || (nodesAlive[i] == false))
                        continue;
                    zmq::message_t outAckMsg(sizeof(ControlMessage));
                    ControlMessage cAckMsg(SUBSCRIPTIONDONE);
                    *((ControlMessage*) outAckMsg.data()) = cAckMsg;
                    controlPublishers[i]->ksend(outAckMsg);
                }
                firstTimeDone = false;
                //fprintf(stderr, "node %u reached here\n", nodeId);
            }

            for(unsigned i=0; i<numNodes; ++i) {
                if((i == nodeId) || (nodesAlive[i] == false))
                    continue;
                //zmq::message_t inMsg;
                if(controlSubscribers[i]->krecv(&inMsg, ZMQ_DONTWAIT)) {
                    ControlMessage cMsg = *((ControlMessage*) inMsg.data());
                    if(cMsg.messageType == SUBSCRIPTIONDONE) {
                        --rem;
                        //fprintf(stderr, "Node %u received SUBSCRIPTIONDONE from %u .. rem = %u\n", nodeId, i, rem);
                    }
                }
            }
            //fprintf(stderr, "node %u rem = %u\n", nodeId, rem);
        } else {
            //fprintf(stderr, "Node %u waiting for %zd items\n", nodeId, topics->size() - subscribed.size());
            /*
               for(it=topics->begin(); it != topics->end(); ++it) {
               if(subscribed.find(*it) == subscribed.end())
               fprintf(stderr, "%u->%u ", nodeId, *it);
               }
             */
            assert(true);
        }
        //NodeManager::barrier(SUBSCRIBE_BARRIER);
        }

        fprintf(stderr, "Node %u: Sending out SUBSCRIPTION_COMPLETE\n", nodeId);
        zmq::message_t outMsg(sizeof(IdType));
        *((IdType*) outMsg.data()) = SUBSCRIPTION_COMPLETE;
        dataPublisher->ksend(outMsg);

        rem = numLiveNodes;
        while(rem > 0) {
            //zmq::message_t inMsg;
            dataSubscriber->krecv(&inMsg);
            IdType mType = *((IdType*) inMsg.data());
            if(mType == SUBSCRIPTION_COMPLETE) {
                --rem;
                //fprintf(stderr, "rem = %u\n", rem);
            }
        }

        //fprintf(stderr, "Node %u: FROM MY END, SUBSCRIPTION IS COMPLETE\n", nodeId);
        NodeManager::barrier(SUBSCRIBEDONE_BARRIER);
        
        pthread_mutex_unlock(&mtx_dataSubscriber);
        pthread_mutex_unlock(&mtx_dataPublisher);

        flushData(numLiveNodes);
    }

    void CommManager::dataPushOut(IdType topic, void* value, unsigned valSize) {
        zmq::message_t outMsg(sizeof(IdType) + valSize);
        *((IdType*) outMsg.data()) = topic;
        memcpy((void*)(((char*) outMsg.data()) + sizeof(IdType)), value, valSize);

        pthread_mutex_lock(&mtx_dataPublisher);
        dataPublisher->ksend(outMsg, ZMQ_DONTWAIT); 
        pthread_mutex_unlock(&mtx_dataPublisher);
    }

    bool CommManager::dataPullIn(IdType &topic, std::vector<int>& value) {
        zmq::message_t inMsg;

        pthread_mutex_lock(&mtx_dataSubscriber);
        bool ret = dataSubscriber->krecv(&inMsg, ZMQ_DONTWAIT);
        pthread_mutex_unlock(&mtx_dataSubscriber);

        if(!ret)
            return false;

	int32_t dataSize = inMsg.size() - sizeof(IdType);
	int32_t numberOfFeatures = dataSize / sizeof(FeatType);
	value.resize(numberOfFeatures);

	memcpy(&topic, inMsg.data(), sizeof(IdType));
        memcpy(value.data(), ((char*)inMsg.data() + sizeof(IdType)), dataSize);
        return true;
    }

    void CommManager::dataSyncPullIn(IdType& topic, std::vector<int>& value) {
        zmq::message_t inMsg;
        pthread_mutex_lock(&mtx_dataSubscriber);
        assert(dataSubscriber->krecv(&inMsg));
        pthread_mutex_unlock(&mtx_dataSubscriber);

	int32_t dataSize = inMsg.size() - sizeof(IdType);
	int32_t numberOfFeatures = dataSize / sizeof(FeatType);
	value.resize(numberOfFeatures);

	memcpy(&topic, inMsg.data(), sizeof(IdType));
	memcpy(value.data(), ((char*)inMsg.data() + sizeof(IdType)), dataSize);
    }

    void CommManager::controlPushOut(unsigned to, void* value, unsigned valSize) {
        assert(to >= 0 && to < numNodes);
        assert(to != nodeId); 
        zmq::message_t outMsg(sizeof(ControlMessage) + valSize);
        *((ControlMessage*) outMsg.data()) = ControlMessage(APPMSG);
        memcpy((void*)(((char*) outMsg.data()) + sizeof(ControlMessage)), value, valSize);

        pthread_mutex_lock(&mtx_controlPublishers[to]);
        assert(controlPublishers[to]->ksend(outMsg, ZMQ_DONTWAIT));
        pthread_mutex_unlock(&mtx_controlPublishers[to]);
    }

    bool CommManager::controlPullIn(unsigned from, void* value, unsigned valSize) {
        assert(from >= 0 && from < numNodes);
        assert(from != nodeId);
        zmq::message_t inMsg;

        pthread_mutex_lock(&mtx_controlSubscribers[from]);
        bool ret = controlSubscribers[from]->krecv(&inMsg, ZMQ_DONTWAIT);
        pthread_mutex_unlock(&mtx_controlSubscribers[from]);

        if(ret == false)
            return false;

        ControlMessage cM;
        cM = *((ControlMessage*) inMsg.data());
        assert(cM.messageType == APPMSG);
        
        /*if(cM.messageType != APPMSG) {
            fprintf(stderr, "CommManager: controlPullIn ignoring message of type %d from %u\n", cM.messageType, from);
            return false;
        }*/

        memcpy(value, ((void*)((char*)inMsg.data() + sizeof(ControlMessage))), valSize);
        return true;
    }

    void CommManager::controlSyncPullIn(unsigned from, void* value, unsigned valSize) {
        assert(from >= 0 && from < numNodes);
        assert(from != nodeId);

        zmq::message_t inMsg;
        //pthread_mutex_lock(&mtx_controlContext);
        pthread_mutex_lock(&mtx_controlSubscribers[from]);
        assert(controlSubscribers[from]->krecv(&inMsg));
        pthread_mutex_unlock(&mtx_controlSubscribers[from]);
        //pthread_mutex_unlock(&mtx_controlContext);

        ControlMessage cM;
        cM = *((ControlMessage*) inMsg.data());
        assert(cM.messageType == APPMSG);
        memcpy(value, ((void*)((char*)inMsg.data() + sizeof(ControlMessage))), valSize);
    }

    void CommManager::flushDataControl() {
        flushControl();
        flushData(numLiveNodes);
    }

    void CommManager::flushData(unsigned nNodes) {
        fprintf(stderr, "Flushing data ...\n");
        //pthread_mutex_lock(&mtx_dataContext);
        pthread_mutex_lock(&mtx_dataPublisher);
        pthread_mutex_lock(&mtx_dataSubscriber);

        zmq::message_t outMsg(sizeof(IdType));
        *((IdType*) outMsg.data()) = NULL_CHAR;
       
        dataPublisher->ksend(outMsg);

        unsigned rem = nNodes;

        while(rem > 0) {
            zmq::message_t inMsg;
            assert(dataSubscriber->krecv(&inMsg));
            IdType idx = *((IdType*) inMsg.data());
            if(idx == NULL_CHAR) {
                --rem;
            } else {
                //fprintf(stderr, "flushing idx = %u\n", idx);
            }
        }

        //pthread_mutex_unlock(&mtx_dataContext);
        pthread_mutex_unlock(&mtx_dataSubscriber);
        pthread_mutex_unlock(&mtx_dataPublisher);
        fprintf(stderr, "Flushing data complete ...\n");
    }

    void CommManager::flushControl() {
        fprintf(stderr, "Flushing control ...\n");
        //pthread_mutex_lock(&mtx_controlContext);

        for(unsigned i=0; i<numNodes; ++i) {
            //fprintf(stderr, "i = %u\n", i);
            if((i == nodeId) || (nodesAlive[i] == false))
                continue;

            pthread_mutex_lock(&mtx_controlPublishers[i]);
            pthread_mutex_lock(&mtx_controlSubscribers[i]);

            zmq::message_t outAckMsg(sizeof(ControlMessage));
            *((ControlMessage*) outAckMsg.data()) = ControlMessage();
            controlPublishers[i]->ksend(outAckMsg);

            while(1) {
                zmq::message_t inMsg;
                if(controlSubscribers[i]->krecv(&inMsg, ZMQ_DONTWAIT)) {
                    ControlMessage cMsg = *((ControlMessage*) inMsg.data());
                    if(cMsg.messageType == NONE)
                        break;
                }
            }
            pthread_mutex_unlock(&mtx_controlSubscribers[i]);
            pthread_mutex_unlock(&mtx_controlPublishers[i]);
        }

        //pthread_mutex_unlock(&mtx_controlContext);
        fprintf(stderr, "Flushing control complete ...\n");
    }

    void CommManager::flushControl(unsigned i) {
        if((i == nodeId) || (nodesAlive[i] == false))
            return;
        fprintf(stderr, "Flushing control %u...\n", i);

        //pthread_mutex_lock(&mtx_controlContext);
        pthread_mutex_lock(&mtx_controlPublishers[i]);
        pthread_mutex_lock(&mtx_controlSubscribers[i]);

        zmq::message_t outAckMsg(sizeof(ControlMessage));
        *((ControlMessage*) outAckMsg.data()) = ControlMessage();
        controlPublishers[i]->ksend(outAckMsg);

        while(1) {
            zmq::message_t inMsg;
            if(controlSubscribers[i]->krecv(&inMsg, ZMQ_DONTWAIT)) {
                ControlMessage cMsg = *((ControlMessage*) inMsg.data());
                if(cMsg.messageType == NONE)
                    break;
            }
        }

        pthread_mutex_unlock(&mtx_controlSubscribers[i]);
        pthread_mutex_unlock(&mtx_controlPublishers[i]);
        //pthread_mutex_unlock(&mtx_controlContext);
    }

    void CommManager::nodeDied(unsigned nId) {
        assert(nodesAlive[nId]);
        nodesAlive[nId] = false;
        --numLiveNodes;
    }
