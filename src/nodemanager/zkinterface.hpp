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
