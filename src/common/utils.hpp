#ifndef __GLOBAL_UTILS_HPP__
#define __GLOBAL_UTILS_HPP__

#include <chrono>
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


/** Feature type is float, so be consistent. */
typedef float FeatType;


static const size_t HEADER_SIZE = sizeof(unsigned) * 5;
enum OP { REQ_FORWARD, PUSH_FORWARD, PULL_FORWARD, REQ_BACKWARD, PUSH_BACKWARD, PULL_BACKWARD, PULL_EVAL, PUSH_EVAL, RESP, INFO, TERM };
enum TYPE { GRAD, AH, Z, ACT, LAB };
enum PROP_TYPE { FORWARD, BACKWARD };

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

// ID represents either layer or data partition, depending on server responding.
static inline void
populateHeader(char* header, unsigned op, unsigned field1 = 0, unsigned field2 = 0, unsigned field3 = 0, unsigned field4 = 0) {
    serialize<unsigned>(header, 0, op);
    serialize<unsigned>(header, 1, field1);
    serialize<unsigned>(header, 2, field2);
    serialize<unsigned>(header, 3, field3);
    serialize<unsigned>(header, 4, field4);
}

static inline unsigned
timestamp_ms() {
    auto now = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count() - 1580333752000ull;
}

static inline void
log(const unsigned nodeId, const char* msg, ...) {
    char* format = new char[strlen(msg) + 1];
    va_list argptr;
    va_start(argptr, msg);
    vsprintf(format, msg, argptr);
    va_end(argptr);

    fprintf(stderr, "\033[1;33m[ Node %2u | %u ]\033[0m %s\n", nodeId, timestamp_ms(), format);

    delete[] format;
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

extern std::mutex writeMutex;
extern std::ofstream debugFile;


#endif // GLOBAL_UTILS_HPP
