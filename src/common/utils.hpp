#ifndef __GLOBAL_UTILS_HPP__
#define __GLOBAL_UTILS_HPP__

#include <chrono>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <sstream>

/** Feature type is float, so be consistent. */
typedef float FeatType;


static const size_t HEADER_SIZE = sizeof(unsigned) * 5;
enum OP { REQ_FORWARD, PUSH_FORWARD, PULL_FORWARD, REQ_BACKWARD, PUSH_BACKWARD, PULL_BACKWARD, PULL_EVAL, PUSH_EVAL, RESP, INFO, TERM };


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

    double getTime() {      // Get floating-point milliseconds.
        std::chrono::duration<double, std::milli> time_span = end - begin;
        return time_span.count();
    }
};


#endif // GLOBAL_UTILS_HPP
