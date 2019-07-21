#include "threadpool.hpp"

ThreadPool::ThreadPool(unsigned nThreads) : numThreads(nThreads), threads(NULL), die(false), running(false) { 
    threads = new ThreadInfo[numThreads]; 
    for(unsigned i=0; i<numThreads; ++i) {
        threads[i].threadId = i;
        threads[i].args = NULL;
        threads[i].thisPtr = this;
    }

    pthread_barrier_init(&bar, NULL, numThreads + 1);
}

ThreadPool::~ThreadPool() {
    delete[] threads;
}

void ThreadPool::createPool() {
    for(unsigned i=0; i<numThreads; ++i) 
        pthread_create(&threads[i].threadHandle, NULL, worker, (void*) &threads[i]);
}

void ThreadPool::destroyPool() {
    die = true;
    pthread_barrier_wait(&bar); // wake up workers

    void* ret;
    for(unsigned i=0; i<numThreads; ++i)
        pthread_join(threads[i].threadHandle, &ret);
    pthread_barrier_destroy(&bar);
}

void ThreadPool::perform(void (*func)(unsigned, void*)) {
    running = true;
    wFunc = func;
    pthread_barrier_wait(&bar); // wake up workers
}

void ThreadPool::sync() {
    if(running) {
        pthread_barrier_wait(&bar);  // sync workers
        running = false;
    }
}

void* ThreadPool::worker(void* args) {
    ThreadInfo* tInfo = (ThreadInfo*) args;
    ThreadPool* thisPtr = tInfo->thisPtr;

    while(1) {
        pthread_barrier_wait(&(thisPtr->bar)); // go to sleep
        if(thisPtr->die) break;

        thisPtr->wFunc(tInfo->threadId, tInfo->args);

        pthread_barrier_wait(&(thisPtr->bar)); // for syncing
    }

    return NULL;
}
