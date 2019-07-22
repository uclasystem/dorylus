#ifndef __COND_HPP__
#define __COND_HPP__


#include <pthread.h>
#include "lock.hpp"


/**
 *
 * Class for a conditional variable.
 * 
 */
class Cond {

private:

    pthread_mutex_t *mLock_ptr;
    pthread_cond_t mCond;

public:

    void init(Lock& mutex_lock) {
        pthread_cond_init(&mCond, NULL);
        mLock_ptr = mutex_lock.internal_ptr();
    }

    void wait() {
        pthread_cond_wait(&mCond, mLock_ptr);
    }

    void signal() {
        pthread_cond_signal(&mCond);
    }

    void destroy() {
        pthread_cond_destroy(&mCond);
    }

    pthread_cond_t *internal_ptr() {
        return &mCond;
    }
};


#endif //__COND_HPP__
