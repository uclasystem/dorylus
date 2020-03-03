#ifndef __GLOBAL_UTILS_HPP__
#define __GLOBAL_UTILS_HPP__

#include <stdarg.h>
#include <stdio.h>
#include <sys/time.h>
#include <cstdlib>

#include <chrono>
#include <climits>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex> 
#include <sstream>
#include <string>
#include <vector>

/** Feature type is float, so be consistent. */
typedef float FeatType;

static const size_t HEADER_SIZE = sizeof(unsigned) * 5;
enum OP {
    REQ_FORWARD,
    PUSH_FORWARD,
    PULL_FORWARD,
    REQ_BACKWARD,
    PUSH_BACKWARD,
    PULL_BACKWARD,
    PULL_EVAL,
    PUSH_EVAL,
    RESP,
    INFO,
    TERM
};
enum TYPE { GRAD, AH, Z, ACT, LAB };
enum PROP_TYPE { FORWARD, BACKWARD };

#define ERR_HEADER_FIELD UINT_MAX

/**
 *
 * Serialization utilities.
 *
 */
template <class T>
static inline void serialize(char* buf, unsigned offset, T val) {
    std::memcpy(buf + (offset * sizeof(T)), &val, sizeof(T));
}

template <class T>
static inline T parse(const char* buf, unsigned offset) {
    T val;
    std::memcpy(&val, buf + (offset * sizeof(T)), sizeof(T));
    return val;
}

// ID represents either layer or data partition, depending on server responding.
static inline void populateHeader(char* header, unsigned op,
                                  unsigned field1 = 0, unsigned field2 = 0,
                                  unsigned field3 = 0, unsigned field4 = 0) {
    serialize<unsigned>(header, 0, op);
    serialize<unsigned>(header, 1, field1);
    serialize<unsigned>(header, 2, field2);
    serialize<unsigned>(header, 3, field3);
    serialize<unsigned>(header, 4, field4);
}

static inline unsigned timestamp_ms() {
    auto now = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               now.time_since_epoch())
               .count() -
           1580333752000ull;
}

static inline void log(const unsigned nodeId, const char* msg, ...) {
    char* format = new char[strlen(msg) + 1];
    va_list argptr;
    va_start(argptr, msg);
    vsprintf(format, msg, argptr);
    va_end(argptr);

    fprintf(stderr, "\033[1;33m[ Node %2u | %u ]\033[0m %s\n", nodeId,
            timestamp_ms(), format);

    delete[] format;
}

static inline void log(std::ofstream& outfile, const char* msg, ...) {
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

    double getTime() {  // Get floating-point milliseconds.
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
using namespace std::chrono;
struct TimerPlus {
    std::chrono::system_clock::time_point begin;
    std::chrono::system_clock::time_point end;
    // vector<mili_duration> durations;
    vector<long long> begin_vec;
    vector<long long> end_vec;
    string name;

    std::mutex timer_mutex;

    TimerPlus() {}
    TimerPlus(const string& name_) { name = name_; }
    void start() { begin = std::chrono::system_clock::now(); }
    void stop() {
        end = std::chrono::system_clock::now();
        std::lock_guard<std::mutex> guard(timer_mutex);
        begin_vec.push_back(
            duration_cast<milliseconds>(begin.time_since_epoch()).count());
        end_vec.push_back(
            duration_cast<milliseconds>(end.time_since_epoch()).count());
        // durations.push_back(end - begin);
    }
    
    void fromString(const std::string& str, unsigned epoch);

    std::string getString(unsigned idx = 0, long long offset = 0) {
        return name + ":" + std::to_string(begin_vec[idx] - offset) + "," +
               std::to_string(end_vec[idx] - offset) + "\n";
    }
};

struct GPUTimers {
    int session;
    ~GPUTimers(){
        for (auto& t : timers) {
            delete getTimer(t.first);
        }
    }
    GPUTimers(){
        std::srand(std::time(nullptr));
        epoch=0;
        session=std::rand()%12345;
    }
    void inc_epoch(){
        epoch++;
    }
    TimerPlus* getTimer(const string& str) {
        if (timers.find(str) == timers.end()) {
            std::lock_guard<std::mutex> guard(timers_mutex);
            timers[str] = new TimerPlus(str);
        }
        return timers[str];
    }

    void report(unsigned idx, unsigned mode) {
        std::string prefix;
        switch (mode) {
            case 0:
                prefix = "LAMBDA";
                break;
            case 1:
                prefix = "GPU";
                break;
            case 2:
                prefix = "CPU";
                break;
        }
        std::ofstream out;
        out.open(prefix + ".ts", std::ios::out);

        long long min = LLONG_MAX;
        for (auto& t : timers) {
            min = std::min(min, t.second->begin_vec[idx]);
        }
        for (auto& t : timers) {
            out  <<prefix <<"_"<< t.second->getString(idx, min);
        }
        out.close();
    }
    std::string lambdaReport() {
        std::string result("");
        for (auto& t : timers) {
            result += t.second->getString();
        }
        return result;
    }
    void addDataStr(const std::string& str);

   private:
    std::map<string, TimerPlus*> timers;
    static std::mutex timers_mutex;
    int epoch;
    
};
extern GPUTimers gtimers;

#endif  // GLOBAL_UTILS_HPP
