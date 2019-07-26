#include <cerrno>
#include <cstring>
#include <cassert>
#include <unistd.h>
#include "zkinterface.hpp"
#include "../utils/utils.hpp"


/** Helper function: convert from type enum to string. */
static const char *
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
static const char *
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


/** Extern class-wide fields. */
zhandle_t *ZKInterface::zh = NULL;
clientid_t ZKInterface::myId;
char *ZKInterface::hostPort = NULL;
char *ZKInterface::clientIdFile = NULL;
Lock ZKInterface::lockWatcher;
Barrier ZKInterface::barInit;


/**
 *
 * Initialize the ZooKeeper interface.
 * 
 */
bool
ZKInterface::init(const char *hPort) {
    hostPort = new char[strlen(hPort) + 1];
    strcpy(hostPort, hPort);

    lockWatcher.init();
    barInit.init(2);

    zh = zookeeper_init(hostPort, watcher, RECV_TIMEOUT, &myId, NULL, 0);
    if (!zh) {
        return false;
    } else {
        barInit.wait();
        return true;
    }
}


/**
 *
 * Free resource for a String_vector struct.
 * 
 */
void
ZKInterface::freeZKStringVector(struct String_vector *v) {
    if (v->data) {
        for (int i = 0; i < v->count; ++i)
            free(v->data[i]);
        free(v->data);
        v->data = 0;
    }
}


/**
 *
 * Create a ZooKeeper node file on given path.
 * 
 */
void
ZKInterface::createZKNode(const char *path, bool ephemeral, bool sync, void (*func) (int, const char *, const void *)) {
    int flags = 0;
    if (ephemeral)
        flags |= ZOO_EPHEMERAL;

    int rc = -1;
    if (sync)
        rc = zoo_create(zh, path, DEFAULT_VALUE, DEFAULT_VALUE_LEN, &ZOO_OPEN_ACL_UNSAFE, flags, NULL, 0);
    else
        rc = zoo_acreate(zh, path, DEFAULT_VALUE, DEFAULT_VALUE_LEN, &ZOO_OPEN_ACL_UNSAFE, flags, func, NULL);

    if (rc)
        printLog(myId.client_id, "ERROR while creating node %s! [Code = %d]\n", path, rc);
}


/**
 *
 * Delete a ZooKeeper node file on given path.
 * 
 */
void
ZKInterface::deleteZKNode(const char *path) {
    assert(zoo_delete(zh, path, -1) == 0);
}


/**
 *
 * Check whether a node file on given path exists.
 * 
 */
bool
ZKInterface::checkZKExists(const char *path, void (*watcherCB)(const char *)) {
    struct Stat stat;
    int rc = zoo_wexists(zh, path, watcher, (void *) watcherCB, &stat);
    return (rc == 0);
}


/**
 *
 * Get the given node's children nodes' info blocks.
 * 
 */
void
ZKInterface::getZKNodeChildren(const char *path, void (*watcherCB) (const char *), struct String_vector *children) {
    assert(zoo_wget_children(zh, path, watcher, (void *) watcherCB, children) == 0);
}


/**
 *
 * Recursively delete all nodes under the given root path.
 * 
 */
int
ZKInterface::recursiveDeleteZKNode(const char *root) {
    struct Stat stat;
    if (zoo_exists(zh, root, 0, &stat) != 0)    
        return 0;

    struct String_vector children;
    int rc = zoo_get_children(zh, root, 0, &children);
    if (rc != ZNONODE) {    // Not reaching the bottom layer yet, so recurse deeper.
        if (rc != ZOK)
            return rc;
        for (int i = 0; i < children.count; ++i) {
            char nodeName[2048];
            snprintf(nodeName, sizeof(nodeName),"%s/%s", root, children.data[i]);
            int rc = recursiveDeleteZKNode(nodeName);
            if (rc != ZOK) {
                freeZKStringVector(&children);
                return rc;
            }
        }
        freeZKStringVector(&children);
    }

    rc = zoo_delete(zh, root, -1);
    return rc;
}


////////////////////////////////////////////////////
// Below are private functions for the interface. //
////////////////////////////////////////////////////


/**
 *
 * Handler for ZooKeeper events.
 * 
 */
void
ZKInterface::watcher(zhandle_t *zzh, int type, int state, const char *path, void *context) {
    lockWatcher.lock();
    if (type == ZOO_SESSION_EVENT)
        handleSessionEvent(zzh, type, state);
    else if(type == ZOO_DELETED_EVENT || type == ZOO_CREATED_EVENT || type == ZOO_CHILD_EVENT)
        handleCreatedDeletedEvent(zzh, type, state, path, context);
    else
        printLog(myId.client_id, "ZK watcher: unknown event type %d. Ignored.\n", type);
    lockWatcher.unlock();
}

void
ZKInterface::handleSessionEvent(zhandle_t *zzh, int type, int state) {
    if (state == ZOO_CONNECTED_STATE) {
        const clientid_t *id = zoo_client_id(zzh);
        if (myId.client_id == 0 || myId.client_id != id->client_id) {
            myId = *id;
            if (clientIdFile) {
                FILE *fh = fopen(clientIdFile, "w");
                if (!fh)
                    perror(clientIdFile);
                else {
                    int rc = fwrite(&myId, sizeof(myId), 1, fh);
                    if (rc != sizeof(myId))
                        perror("writing client id");
                    fclose(fh);
                }
            }
            barInit.wait();
        }
    } else if (state == ZOO_AUTH_FAILED_STATE) {
        printLog(myId.client_id, "ZK session: authentication failure. Shutting down...\n");
        zookeeper_close(zzh);
        zh = 0;
    } else if (state == ZOO_EXPIRED_SESSION_STATE) {
        printLog(myId.client_id, "ZK session: session time expired. Shutting down...\n");
        zookeeper_close(zzh);
        zh = 0;
    } else
        printLog(myId.client_id, "ZK session: unknown session event \"%s\". Ignored.\n", state2String(state));
}

void
ZKInterface::handleCreatedDeletedEvent(zhandle_t *zzh, int type, int state, const char *path, void *context) {
    assert(context != NULL);
    ((void (*)(const char *)) context)(path);
}
