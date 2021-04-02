#ifndef __GLOBAL_UTILS_HPP__
#define __GLOBAL_UTILS_HPP__

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <map>

#include <mutex> // for debugging, delete later

#include <stdio.h>
#include <stdarg.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <unistd.h>
#include <limits.h>


/** Feature type is float, so be consistent. */
typedef float FeatType;
typedef float EdgeType;

#define HEADER_SIZE (sizeof(unsigned) + sizeof(Chunk)) // sizeof(OP) + sizeof(Chunk)
// OP, TENSOR_NAME, FIELD0, FIELD1, ...
static const size_t TENSOR_NAME_SIZE = 8;
static const size_t TENSOR_HDR_SIZE = sizeof(unsigned) * 5 + TENSOR_NAME_SIZE;
enum OP {
    REQ_VTX_FORWARD, PUSH_VTX_FORWARD, PULL_VTX_FORWARD,
    REQ_VTX_BACKWARD, PUSH_VTX_BACKWARD, PULL_VTX_BACKWARD,
    PULL_VTX_EVAL, PUSH_VTX_EVAL,
    REQ_EDG_FORWARD, PUSH_EDG_FORWARD, PULL_EDG_FORWARD,
    REQ_EDG_BACKWARD, PUSH_EDG_BACKWARD, PULL_EDG_BACKWARD,
    PULL_EDG_EVAL, PUSH_EDG_EVAL,
    PUSH, PULL, PULLE, PUSHE, PULLEINFO, FIN, EVAL,
    RESP, INFO, TERM
};
enum TYPE { GRAD, AH, Z, ACT, LAB };
enum PROP_TYPE { FORWARD, BACKWARD };
enum AGGREGATOR { WSUM, MEAN, ADD, MIN, MAX };
enum GNN { GCN, GAT };

enum CONVERGE_STATE { EARLY, CLOSE, DONE, FLUCT, NUM_STATE };
static std::string CONVERGE_STATE_STR[CONVERGE_STATE::NUM_STATE] = {
    "EARLY", "CLOSE", "DONE", "FLUCT"
};

#define ERR_HEADER_FIELD UINT_MAX
#define NOT_FOUND_ERR_FIELD (UINT_MAX - 1)
#define DUPLICATE_REQ_ERR_FIELD (UINT_MAX - 2)
#define CHUNK_DNE_ERR (UINT_MAX - 3)

#define TRAIN_PORTION 0.66
#define VAL_PORTION 0.1
#define TEST_PORTION 0.24

struct Chunk {
    unsigned localId;
    unsigned globalId;
    unsigned lowBound;
    unsigned upBound;
    unsigned layer;
    PROP_TYPE dir;

    unsigned epoch;

    bool vertex;

    bool operator<(const Chunk &rhs) const {
        // TODO: (YIFAN) Assign priority in the computation sequence
        return
            epoch > rhs.epoch || (epoch == rhs.epoch && (
            dir > rhs.dir || (dir == rhs.dir && (
            (dir == PROP_TYPE::FORWARD && layer > rhs.layer) ||
            (dir == PROP_TYPE::BACKWARD && layer < rhs.layer) || (layer == rhs.layer && (
            (dir == PROP_TYPE::FORWARD && !vertex && rhs.vertex) ||
            (dir == PROP_TYPE::BACKWARD && vertex && !rhs.vertex) || (vertex == rhs.vertex && (
            localId > rhs.localId || (localId == rhs.localId && (
            globalId > rhs.globalId || (globalId == rhs.globalId && (
            lowBound > rhs.lowBound || (lowBound == rhs.lowBound && (
            upBound > rhs.upBound))))))))))))));
    }

    std::string str() const {
        char buf[128];
        sprintf(buf, "%u:%s:%u:%u/%u: vtx %u", epoch, dir == PROP_TYPE::FORWARD ? "F" : "B",
          layer, localId, globalId, vertex);

        return std::string(buf);
    }

    bool isFirstLayer() {
        return dir == PROP_TYPE::FORWARD && layer == 0 && vertex;
    }
    bool isLastLayer() {
        return dir == PROP_TYPE::BACKWARD && layer == 0 && vertex;
    }
};

Chunk createChunk(unsigned rows, unsigned nChunks, unsigned id, unsigned layer, PROP_TYPE dir, unsigned ep = 0, bool vertex = true);

// backoff sleep strategy to improve CPU utilization
struct BackoffSleeper {
    unsigned trails = 0;
    unsigned failedTrials = 0;
    const int INIT_PERIOD = 1024;
    const int MAX_PERIOD = 16384;
    int SLEEP_PERIOD = INIT_PERIOD;

    void sleep() {
        usleep(SLEEP_PERIOD);
        failedTrials++;
        if (failedTrials == 64 && SLEEP_PERIOD < MAX_PERIOD) {
            failedTrials = 0;
            SLEEP_PERIOD *= 2;
        }
    }

    void reset() {
        SLEEP_PERIOD = INIT_PERIOD;
        failedTrials = 0;
        trails = 0;
    }
};

inline size_t argmax(FeatType* first, FeatType* last) { return std::distance(first, std::max_element(first, last)); }

inline size_t getFileSize(const char* fname) {
    struct stat st;
    if (stat(fname, &st) != 0) {
        return 0;
    }

    return st.st_size;
}


/**
 *
 * Serialization utilities.
 *
 */
template<class T>
static inline void
serialize(char *buf, unsigned offset, T val) {
    std::memcpy(buf + (offset * sizeof(T)), &val, sizeof(T));
}

template<class T>
static inline T
parse(const void* data, unsigned offset) {
    char* buf = (char*)data;
    T val;
    std::memcpy(&val, buf + (offset * sizeof(T)), sizeof(T));
    return val;
}

template<class T>
static inline T
parse(const char *buf, unsigned offset) {
    T val;
    std::memcpy(&val, buf + (offset * sizeof(T)), sizeof(T));
    return val;
}

static inline std::string
parseName(const void* data) {
    char* buf = (char*)data;
    return std::string(buf + sizeof(unsigned));
}

static inline std::string
parseName(const char* buf) {
    return std::string(buf + sizeof(unsigned));
}

// For summing standard tensors
static inline float
sumTensor(unsigned rows, unsigned cols, FeatType* tensor) {
    unsigned items = rows * cols;
    float sum = 0;
    for (unsigned u = 0; u < items; ++u) {
        sum += tensor[u];
    }

    return sum;
}

// For summing edge tensors
static inline float
sumTensor(unsigned rows, unsigned cols, FeatType** tensor) {
    float sum = 0;
    for (unsigned ur = 0; ur < rows; ++ur) {
        for (unsigned uc = 0; uc < cols; ++uc) {
            sum += tensor[ur][uc];
        }
    }

    return sum;
}

// ID represents either layer or data partition, depending on server responding.
static inline void
populateHeader(char* header, unsigned op, unsigned field1 = 0, unsigned field2 = 0, unsigned field3 = 0, unsigned field4 = 0) {
    serialize<unsigned>(header, 0, op);
    serialize<unsigned>(header, 1, field1);
    serialize<unsigned>(header, 2, field2);
    serialize<unsigned>(header, 3, field3);
    serialize<unsigned>(header, 4, field4);
}

static inline void
populateHeader(void* ptr, unsigned op, unsigned field1 = 0, unsigned field2 = 0, unsigned field3 = 0, unsigned field4 = 0) {
    char* header = (char*)ptr;
    serialize<unsigned>(header, 0, op);
    serialize<unsigned>(header, 1, field1);
    serialize<unsigned>(header, 2, field2);
    serialize<unsigned>(header, 3, field3);
    serialize<unsigned>(header, 4, field4);
}

static inline void
populateHeader(void* header, unsigned op, const char* tensorName, unsigned field1 = 0,
  unsigned field2 = 0, unsigned field3 = 0, unsigned field4 = 0) {
    char* data = (char*)header;
    serialize<unsigned>(data, 0, op);
    std::memcpy(data + sizeof(unsigned), tensorName, TENSOR_NAME_SIZE);
    serialize<unsigned>(data, 3, field1);
    serialize<unsigned>(data, 4, field2);
    serialize<unsigned>(data, 5, field3);
    serialize<unsigned>(data, 6, field4);
}

static inline unsigned
timestamp_ms() {
    using namespace std::chrono;
    auto now = high_resolution_clock::now();
    return duration_cast<milliseconds>(now.time_since_epoch()).count() % (1 << 30);
}



static inline void
log(std::ofstream& outfile, const char *msg, ...) {
    char format[strlen(msg) + 1];
    va_list argptr;
    va_start(argptr, msg);
    vsprintf(format, msg, argptr);
    va_end(argptr);

    size_t msgSize = 12 + strlen(format);
    char logMsg[msgSize];
    sprintf(logMsg, "%u %s\n", timestamp_ms(), format);

    outfile.write(logMsg, msgSize);
}

/**
 *
 * Struct for a timer.
 *
 */
struct Timer {
    std::chrono::high_resolution_clock::time_point begin;
    std::chrono::high_resolution_clock::time_point end;

    void start() { begin = std::chrono::high_resolution_clock::now(); }
    void stop() { end = std::chrono::high_resolution_clock::now(); }

    double peek() {
        auto now = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> time_span = now - begin;
        return time_span.count();
    }

    double getTime() {      // Get floating-point milliseconds.
        std::chrono::duration<double, std::milli> time_span = end - begin;
        return time_span.count();
    }
};

/**
 *
 * Struct for a timer.
 *
 */
typedef std::chrono::duration<double, std::milli> mili_duration;
using std::string;
using std::vector;
struct TimerPlus {
    std::chrono::high_resolution_clock::time_point begin;
    std::chrono::high_resolution_clock::time_point end;
    vector<mili_duration> durations;
    string name;

    TimerPlus() {}
    TimerPlus(const string& name_) {name = name_;}
    void start() {
        begin = std::chrono::high_resolution_clock::now();
    }
    void stop() {
        end = std::chrono::high_resolution_clock::now();
        durations.push_back(end - begin);
    }
    void report() {
        mili_duration max_d = mili_duration::zero();
        mili_duration avg_d = mili_duration::zero();
        mili_duration total_d = mili_duration::zero();
        for (size_t i = 0; i < durations.size(); ++i) {
            total_d += durations[i];
            max_d = max(max_d, durations[i]);
        }
        avg_d = total_d / durations.size();
        std::cout << name + "Timer : \n";
        std::cout << "Max: " << max_d.count() << "ms \n";
        std::cout << "Avg: " << avg_d.count() << "ms \n";
        std::cout << "Tot: " << total_d.count() << "ms \n";
    }
};


struct GPUTimers {
    TimerPlus* getTimer(const string& str) {
        if (timers.find(str) == timers.end())
            timers[str] = new TimerPlus(str);
        return timers[str];
    }
    void report() {
        for (auto & t : timers) {
            t.second->report();
        }
    }
private:
    std::map<string, TimerPlus*> timers;
};
extern GPUTimers gtimers;

extern std::ofstream debugFile;
extern std::mutex fileMutex;

extern FILE* outputFile;

extern std::ofstream matrixFile;
extern std::mutex mFileMutex;

void matrixToFile(FeatType* fptr, unsigned start, unsigned end, unsigned c);

static inline void
sleep_ms(unsigned t) {
    std::this_thread::sleep_for(std::chrono::milliseconds(t));
}

static inline void
log(const unsigned nodeId, const char *format, ...) {
    va_list args;
    va_start(args, format);
    vfprintf(outputFile, format, args);
    va_end(args);

    fputc('\n', outputFile);

    fflush(outputFile);
}

#endif // GLOBAL_UTILS_HPP
