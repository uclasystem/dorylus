#ifndef __THREAD_POOL_H__
#define __THREAD_POOL_H__

#include <pthread.h>

class ThreadPool;

typedef struct threadInfo {
    pthread_t threadHandle;
    unsigned threadId;
    ThreadPool* thisPtr;
    void* args;
} ThreadInfo;

class ThreadPool {
public:
    ThreadPool(unsigned nThreads);
    ~ThreadPool();
    void createPool();
    void destroyPool();

    void perform(void (*func)(unsigned, void*));
    void sync();

private:
    unsigned numThreads;
    ThreadInfo* threads;
    pthread_barrier_t bar;
    bool die;
    bool running;

    void (*wFunc)(unsigned, void*);
    static void* worker(void* args);
};

#endif //__THREAD_POOL_H__
