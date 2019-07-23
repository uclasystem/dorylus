#ifndef __LOCK_HPP__
#define __LOCK_HPP__


#include <pthread.h>


/**
 *
 * Class of a mutex lock.
 * 
 */
class Lock {

private:

    pthread_mutex_t mLock;

public:

    void init() {
        pthread_mutex_init(&mLock, NULL);
    }

    void lock() {
        pthread_mutex_lock(&mLock);
    }

    void unlock() {
        pthread_mutex_unlock(&mLock);
    }

    void destroy() {
        pthread_mutex_destroy(&mLock);
    }

    pthread_mutex_t *internal_ptr() {
        return &mLock;
    }
};


#endif //__LOCK_HPP__
