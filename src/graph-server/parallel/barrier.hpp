#ifndef __BARRIER_HPP__
#define __BARRIER_HPP__


#include <cassert>
#include <pthread.h>


/**
 *
 * Class of a cross-thread barrier (for shared-memory, on a single node).
 * 
 */
class Barrier {

private:

    pthread_barrier_t bar;

public:

    void init(unsigned n = 0) {
        assert(n > 0);
        pthread_barrier_init(&bar, NULL, n);
    }

    void wait() {
        pthread_barrier_wait(&bar);
    }

    void destroy() {
        pthread_barrier_destroy(&bar);
    }
};


#endif //__BARRIER_HPP__
