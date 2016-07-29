// NOTE: This r-barrier is taken from existing external project.
// Contact Keval Vora: kvora001@cs.ucr.edu for more details. 
// Be careful before using it in production and ensure that it adheres to the requried license.
 
#ifndef __R_BARRIER_HPP__
#define __R_BARRIER_HPP__

#include <cassert>
#include <pthread.h>

// TODO: Optimize this implementation
class RBarrier {
    pthread_mutex_t mtxWaiter;
    pthread_cond_t condWaiter;
    unsigned nWaiters;
    unsigned nRemaining;

public:
    void init(unsigned n = 0) {
        assert(n > 0);
        nWaiters = n;
        nRemaining = nWaiters;
        pthread_mutex_init(&mtxWaiter, NULL);
        pthread_cond_init(&condWaiter, NULL);
    }

    void wait() {
        pthread_mutex_lock(&mtxWaiter);
        --nRemaining;
        assert(nRemaining >= 0);
        if(nRemaining != 0) {
            pthread_cond_wait(&condWaiter, &mtxWaiter);
        } else {
            pthread_cond_broadcast(&condWaiter);
            nRemaining = nWaiters;
        }
        pthread_mutex_unlock(&mtxWaiter);
    }

    void releaseAll(){
        pthread_mutex_lock(&mtxWaiter);
        if((nRemaining != 0) && (nRemaining != nWaiters)) {
            pthread_cond_broadcast(&condWaiter);
        }
        nRemaining = nWaiters;
        pthread_mutex_unlock(&mtxWaiter);
    }

    unsigned numBlockedThreads() {
        pthread_mutex_lock(&mtxWaiter);
        unsigned ret = nRemaining;
        pthread_mutex_unlock(&mtxWaiter);
        return ret;
    }

    void destroy() {
        pthread_mutex_destroy(&mtxWaiter);
        pthread_cond_destroy(&condWaiter);
    }
};

#endif //__R_BARRIER_HPP__
