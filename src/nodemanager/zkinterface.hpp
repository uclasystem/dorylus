#ifndef __ZK_INTERFACE_HPP__
#define __ZK_INTERFACE_HPP__


#include <zookeeper.h>
#include <iostream>
#include <pthread.h>
#include "../parallel/lock.hpp"
#include "../parallel/barrier.hpp"


#define RECV_TIMEOUT 1000


#define DEFAULT_VALUE ""
#define DEFAULT_VALUE_LEN 0


/** Helper function: convert from type enum to string. */
const char *
type2String(int state) {
    if (state == ZOO_CREATED_EVENT)
        return "CREATED_EVENT";
    if (state == ZOO_DELETED_EVENT)
        return "DELETED_EVENT";
    if (state == ZOO_CHANGED_EVENT)
        return "CHANGED_EVENT";
    if (state == ZOO_CHILD_EVENT)
        return "CHILD_EVENT";
    if (state == ZOO_SESSION_EVENT)
        return "SESSION_EVENT";
    if (state == ZOO_NOTWATCHING_EVENT)
        return "NOTWATCHING_EVENT";

    return "UNKNOWN_EVENT_TYPE";
}


/** Helper function: convert from state enum to string. */
const char *
state2String(int state) {
    if (state == 0)
        return "CLOSED_STATE";
    if (state == ZOO_CONNECTING_STATE)
        return "CONNECTING_STATE";
    if (state == ZOO_ASSOCIATING_STATE)
        return "ASSOCIATING_STATE";
    if (state == ZOO_CONNECTED_STATE)
        return "CONNECTED_STATE";
    if (state == ZOO_EXPIRED_SESSION_STATE)
        return "EXPIRED_SESSION_STATE";
    if (state == ZOO_AUTH_FAILED_STATE)
        return "AUTH_FAILED_STATE";

    return "INVALID_STATE";
}


/**
 *
 * Class of ZooKeeper interface, wrapping over ZooKeeper API.
 * 
 */
class ZKInterface {

public:

    static bool init(const char *hPort);
    static void freeZKStringVector(struct String_vector *v);
    static void createZKNode(const char *path, bool ephemeral, bool sync, void (*func) (int, const char *, const void *));
    static void deleteZKNode(const char *path);
    static bool checkZKExists(const char *path, void (* watcherCB)(const char *));
    static void getZKNodeChildren(const char *path, void (*watcherCB) (const char *), struct String_vector *children);
    static int recursiveDeleteZKNode(const char *root);

private:

    static zhandle_t* zh;
    static clientid_t myId;
    static char *hostPort;
    static char *clientIdFile; // TODO: Remove this variable.
    static Lock lockWatcher;
    static Barrier barInit;

    static void watcher(zhandle_t *zzh, int type, int state, const char *path, void *context);
    static void handleSessionEvent(zhandle_t *zzh, int type, int state);
    static void handleCreatedDeletedEvent(zhandle_t *zzh, int type, int state, const char *path, void *context);
};


#endif //__ZK_INTERFACE_HPP__
