#ifndef __GLOBAL_UTILS_HPP__
#define __GLOBAL_UTILS_HPP__

#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>
#include <map>

#include <mutex> // for debugging, delete later

#include <stdio.h>
#include <stdarg.h>
#include <sys/time.h>
#include <limits.h>


/** Feature type is float, so be consistent. */
typedef float FeatType;

#define TENSOR_NAME_SIZE 8

// base timestamp for profiling
#define BASE_TMSP (1580333752000ull)

const size_t HEADER_SIZE = sizeof(unsigned) * 5 + sizeof(unsigned) * 2;
// OP, TENSOR_NAME, FIELD0, FIELD1, ...
static const size_t TENSOR_HDR_SIZE = sizeof(unsigned) * 5 + TENSOR_NAME_SIZE;
enum OP {
    REQ_VTX_FORWARD, PUSH_VTX_FORWARD, PULL_VTX_FORWARD,
    REQ_VTX_BACKWARD, PUSH_VTX_BACKWARD, PULL_VTX_BACKWARD,
    PULL_VTX_EVAL, PUSH_VTX_EVAL,
    REQ_EDG_FORWARD, PUSH_EDG_FORWARD, PULL_EDG_FORWARD,
    REQ_EDG_BACKWARD, PUSH_EDG_BACKWARD, PULL_EDG_BACKWARD,
    PULL_EDG_EVAL, PUSH_EDG_EVAL,
    PUSH, PULL, FIN,
    RESP, INFO, TERM
};
enum TYPE { GRAD, AH, Z, ACT, LAB };
enum PROP_TYPE { FORWARD, BACKWARD };
enum AGGREGATOR { WSUM, MEAN, ADD, MIN, MAX };

#define ERR_HEADER_FIELD UINT_MAX


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
parse(const char *buf, unsigned offset) {
    T val;
    std::memcpy(&val, buf + (offset * sizeof(T)), sizeof(T));
    return val;
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
    return duration_cast<milliseconds>(now.time_since_epoch()).count() - BASE_TMSP;
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
log(const unsigned nodeId, const char *format, ...) {
    va_list args;
    va_start(args, format);
    vfprintf(outputFile, format, args);
    va_end(args);

    fputc('\n', outputFile);

    fflush(outputFile);
}

#endif // GLOBAL_UTILS_HPP
