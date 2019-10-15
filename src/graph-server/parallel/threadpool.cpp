#include "threadpool.hpp"


ThreadPool::ThreadPool(unsigned nThreads)
    : numThreads(nThreads), threads(NULL), die(false), running(false) {

    // Initialize a bunch of thread info blocks.
    threads = new ThreadInfo[numThreads];
    for (unsigned i = 0; i < numThreads; ++i) {
        threads[i].threadId = i;
        threads[i].args = NULL;
        threads[i].thisPtr = this;
    }

    pthread_barrier_init(&bar, NULL, numThreads + 1);
}

ThreadPool::~ThreadPool() {
    delete[] threads;
}


/**
 *
 * Create actual running pthreads.
 *
 */
void
ThreadPool::createPool() {
    for (unsigned i = 0; i < numThreads; ++i)
        pthread_create(&threads[i].threadHandle, NULL, worker, (void *) &threads[i]);
}


/**
 *
 * Destroy the thread pool.
 *
 */
void
ThreadPool::destroyPool() {
    die = true;
    pthread_barrier_wait(&bar);     // Wake up workers.

    void *ret;
    for (unsigned i = 0; i < numThreads; ++i)
        pthread_join(threads[i].threadHandle, &ret);

    pthread_barrier_destroy(&bar);
}


/**
 *
 * Send all the workers in the pool to perform the task on the given function.
 *
 */
void
ThreadPool::perform(std::function<void(unsigned, void *)> func) {
    running = true;
    wFunc = func;
    pthread_barrier_wait(&bar);     // Wake up workers.
}

void
ThreadPool::perform(std::function<void(unsigned, void *)> func, void *args) {
    for (unsigned i = 0; i < numThreads; ++i) {
        threads[i].args = args;
    }
    running = true;
    wFunc = func;
    pthread_barrier_wait(&bar);
}

/**
 *
 * Join all the thread in the pool.
 *
 */
void
ThreadPool::sync() {
    if (running) {
        pthread_barrier_wait(&bar);     // Sync workers.
        running = false;
    }
}


/**
 *
 * The worker function. Blocks on the first barrier when created, then wakes up on `perform()`
 * and blocks on the second barrier until all workers finish. A rather clever design.
 *
 */
void *
ThreadPool::worker(void *args) {
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
