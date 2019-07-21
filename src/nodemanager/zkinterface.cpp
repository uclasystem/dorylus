#include "zkinterface.hpp"
#include "../utils/utils.hpp"
#include <cerrno>
#include <cstring>
#include <cassert>
#include <unistd.h>

zhandle_t* ZKInterface::zh = NULL;
clientid_t ZKInterface::myId;
char* ZKInterface::hostPort = NULL;
char* ZKInterface::clientIdFile = NULL;
pthread_mutex_t ZKInterface::mtx_watcher;
pthread_barrier_t ZKInterface::bar_init;

int ZKInterface::freeZKStringVector(struct String_vector *v) {
    if (v->data) {
        int i;
        for(i=0;i<v->count; i++) {
            free(v->data[i]);
        }
        free(v->data);
        v->data = 0;
    }
    return 0;
}

bool ZKInterface::init(const char* hPort) {
    hostPort = new char[strlen(hPort) + 1];
    strcpy(hostPort, hPort);

    pthread_mutex_init(&mtx_watcher, NULL);
    pthread_barrier_init(&bar_init, NULL, 2);

    zh = zookeeper_init(hostPort, watcher, RECV_TIMEOUT, &myId, NULL, 0);
    if(!zh) {
        efprintf(stderr, "Error initializing zookeeper service! code: %d, string: %s\n", errno, strerror(errno));
        return false;
    }
    pthread_barrier_wait(&bar_init);
    return true;
}

void ZKInterface::watcher(zhandle_t* zzh, int type, int state, const char* path, void* context) {
    pthread_mutex_lock(&mtx_watcher);
/*  
    fprintf(stderr, "Watcher %s state = %s", type2String(type), state2String(state));
    if (path && strlen(path) > 0) {
        fprintf(stderr, " for path %s", path);
    }
    fprintf(stderr, "\n");
*/
    if(type == ZOO_SESSION_EVENT) {
        handleSessionEvent(zzh, type, state);
    } else if(type == ZOO_DELETED_EVENT || type == ZOO_CREATED_EVENT || type == ZOO_CHILD_EVENT) {
        handleCreatedDeletedEvent(zzh, type, state, path, context);
    } else {
        fprintf(stderr, "Watcher event type %s not handled.\n", type2String(type));
    }
    pthread_mutex_unlock(&mtx_watcher);
}

void ZKInterface::handleSessionEvent(zhandle_t* zzh, int type, int state) {
    if(state == ZOO_CONNECTED_STATE) {
        const clientid_t *id = zoo_client_id(zzh);
        if (myId.client_id == 0 || myId.client_id != id->client_id) {
            myId = *id;
            fprintf(stderr, "Got a new session id: 0x%llx\n",
                    (long long) myId.client_id);
            if (clientIdFile) {
                FILE *fh = fopen(clientIdFile, "w");
                if (!fh) {
                    perror(clientIdFile);
                } else {
                    int rc = fwrite(&myId, sizeof(myId), 1, fh);
                    if (rc != sizeof(myId)) {
                        perror("writing client id");
                    }
                    fclose(fh);
                }
            }
            pthread_barrier_wait(&bar_init);
        }
    } else if(state == ZOO_AUTH_FAILED_STATE) {
        fprintf(stderr, "Authentication failure. Shutting down...\n");
        zookeeper_close(zzh);
        zh=0;
    } else if(state == ZOO_EXPIRED_SESSION_STATE) {
        fprintf(stderr, "Session expired. Shutting down...\n");
        zookeeper_close(zzh);
        zh=0;
    } else {
        fprintf(stderr, "Session event state %s not handled.\n", state2String(state));
    }
}

void ZKInterface::handleCreatedDeletedEvent(zhandle_t* zzh, int type, int state, const char* path, void* context) {
    assert(context != NULL);
    ((void (*)(const char*)) context)(path);
}

void ZKInterface::createZKNode(const char* path, bool ephemeral, bool sync, void (*func) (int, const char*, const void*)) {
    int flags = 0;
    if(ephemeral)
        flags |= ZOO_EPHEMERAL;

    int rc = -1;
    if(sync)
        rc = zoo_create(zh, path, DEFAULT_VALUE, DEFAULT_VALUE_LEN, &ZOO_OPEN_ACL_UNSAFE, flags, NULL, 0); // DEFAULT_VALUE, DEFAULT_VALUE_LEN);
    else
        rc = zoo_acreate(zh, path, DEFAULT_VALUE, DEFAULT_VALUE_LEN, &ZOO_OPEN_ACL_UNSAFE, flags, func, NULL);

    if (rc)
        fprintf(stderr, "Error %d while creating %s\n", rc, path);
}

void ZKInterface::deleteZKNode(const char* path) {
    assert(zoo_delete(zh, path, -1) == 0);
}

bool ZKInterface::checkZKExists(const char* path, void (*watcherCB)(const char*)) {
    struct Stat stat;
    int rc = zoo_wexists(zh, path, watcher, (void *) watcherCB, &stat);
    return (rc == 0);
}

void ZKInterface::getZKNodeChildren(const char* path, void (*watcherCB) (const char*), struct String_vector* children) {
    assert(zoo_wget_children(zh, path, watcher, (void *) watcherCB, children) == 0);
}

int ZKInterface::recursiveDeleteZKNode(const char* root) {
    struct Stat stat;
    if(zoo_exists(zh, root, 0, &stat) != 0) {
        fprintf(stderr, "Node %s does not exist\n", root);
        return 0;
    }

    struct String_vector children;
    int i;
    int rc = zoo_get_children(zh, root, 0, &children);
    if(rc != ZNONODE){
        if(rc != ZOK){
            fprintf(stderr, "Failed to get children of %s, rc=%d\n", root, rc);
            return rc;
        }
        for(i=0;i<children.count; i++){
            int rc = 0;
            char nodeName[2048];
            snprintf(nodeName, sizeof(nodeName),"%s/%s", root, children.data[i]);
            rc=recursiveDeleteZKNode(nodeName);
            if(rc!=ZOK){
                freeZKStringVector(&children);
                return rc;
            }
        }
        freeZKStringVector(&children);
    }
    fprintf(stderr, "Deleting %s\n", root);
    rc = zoo_delete(zh,root,-1);
    if(rc != ZOK){
        fprintf(stderr, "Failed to delete znode %s, rc=%d\n", root, rc);
    } 
    return rc;
}

