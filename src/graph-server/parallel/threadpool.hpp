#ifndef __THREAD_POOL_HPP__
#define __THREAD_POOL_HPP__


#include <pthread.h>
#include <functional>


/** Structure of a thread's info block. */
class ThreadPool;

typedef struct threadInfo {
    pthread_t threadHandle;
    unsigned threadId;
    ThreadPool *thisPtr;
    void *args;
} ThreadInfo;


/**
 *
 * Class of a thread pool on a node.
 *
 */
class ThreadPool {

public:

    ThreadPool(unsigned nThreads);
    ~ThreadPool();

    void createPool();
    void destroyPool();

    void perform(std::function<void(unsigned, void *)> func);
    void perform(std::function<void(unsigned, void *)> func, void *args);
    void sync();

private:

    unsigned numThreads;
    ThreadInfo *threads;

    pthread_barrier_t bar;

    bool die;
    bool running;

    std::function<void(unsigned, void *)> wFunc;
    static void *worker(void *args);
};


#endif //__THREAD_POOL_HPP__
