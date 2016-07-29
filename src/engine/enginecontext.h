#ifndef __ENGINE_CONTEXT_H__
#define __ENGINE_CONTEXT_H__

#include "bitsetscheduler.hpp"
#include "densebitset.hpp"

class EngineContext {
public:
    void signal(IdType vId) {
        scheduler->schedule(vId);
    }

    void setScheduler(BitsetScheduler* sched) {
        scheduler = sched;
    }

    IdType numVertices() {
      return nVertices;
    }

    void setNumVertices(IdType nv) {
      nVertices = nv;
    }

    unsigned currentBatch() {
      return cBatch;
    }

    void setCurrentBatch(unsigned cb) {
      cBatch = cb;
    }

    void setTooLong(bool tl) {
      tLong = tl;
    }

    bool tooLong() {
      return tLong;
    }

private:
    BitsetScheduler* scheduler;
    IdType nVertices;
    unsigned cBatch;
    bool tLong;
};

#endif //__ENGINE_CONTEXT_H__
