#include <cmath>
#include <fstream>
#include <iostream>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/time.h>
#include <cstring>
#include <cassert>
#include "utils.hpp"


/**
 *
 * Print a log message to the log file.
 *
 */
void
printLog(const unsigned nodeId, const char *msg, ...) {

    // Plug in the node ID.
    // char format[16 + strlen(msg)];
    char format[16 + 1024];
    sprintf(format, "[ Node %3u ]  %s\n", nodeId, msg);

    // Print the log message.
    va_list argptr;
    va_start(argptr, msg);
    vfprintf(stderr, format, argptr);
    va_end(argptr);
}


/**
 *
 * Get current timer value.
 *
 */
double getTimer() {
    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec * 1000 + t.tv_usec / 1000.0;
}

/**
 *
 * Get current date and time
 *
 */
std::time_t getCurrentTime() {
    auto time_now = std::chrono::system_clock::now();

    std::time_t current_time = std::chrono::system_clock::to_time_t(time_now);
    return current_time;
}

/**
 *
 * Read in ip address from file.
 *
 */
void
getPrIP(std::string& myPrIpFile, std::string& ip) {
    std::ifstream ipFile(myPrIpFile);
    assert(ipFile.good());

    std::getline(ipFile, ip);

    ipFile.close();
}
