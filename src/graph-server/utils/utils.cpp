#include <fstream>
#include <iostream>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/time.h>
#include <net/if.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <cstring>
#include <cassert>
#include "utils.hpp"


/**
 *
 * Print a log message to the log file.
 * 
 */
void
printLog(const unsigned nodeId, const char *format, ...) {

    // Print node ID.
    fprintf(stderr, "[ Node %u ] ", nodeId);

    // Print the log message.
    va_list argptr;
    va_start(argptr, format);
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



void
getPrIP(std::string& myPrIpFile, std::string& ip) { 
    std::ifstream ipFile(myPrIpFile);
    assert(ipFile.good());

    std::getline(ipFile, ip);

    ipFile.close();
}

void
getPubIP(std::string& myPubIpFile, std::string& ip) {
    std::ifstream ipFile(myPubIpFile);
    assert(ipFile.good());

    std::getline(ipFile, ip);

    ipFile.close();
}
