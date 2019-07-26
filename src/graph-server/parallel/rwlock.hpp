#ifndef __RWLOCK_HPP__
#define __RWLOCK_HPP__


#include <pthread.h>


/**
 *
 * Class of a reader-writer lock.
 * 
 */
class RWLock {

private:

    pthread_rwlock_t rwlock;

public:
    void init() {
        pthread_rwlock_init(&rwlock, NULL);
    }

    void readLock() {
        pthread_rwlock_rdlock(&rwlock);
    }

    void writeLock() {
        pthread_rwlock_wrlock(&rwlock);
    }

    void unlock() {
        pthread_rwlock_unlock(&rwlock);
    }

    void destroy() {
        pthread_rwlock_destroy(&rwlock);
    }
};


#endif //__RWLOCK_HPP__
