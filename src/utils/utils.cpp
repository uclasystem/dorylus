#include <fstream>
#include <iostream>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/time.h>
#include <net/if.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <cstring>
#include "utils.hpp"


std::map<size_t, std::string> typeToFormatSpecifier;


void initArgs() {
    typeToFormatSpecifier[typeid(int).hash_code()] = "%d";
    typeToFormatSpecifier[typeid(float).hash_code()] = "%f";
    typeToFormatSpecifier[typeid(IdType).hash_code()] = "%llu";
    typeToFormatSpecifier[typeid(char).hash_code()] = "%s"; 
}


void removeArg(int *argc, char **argv, int i) {
    if (i < *argc)
        (*argc)--;
    for (; i < *argc; i++)
        argv[i] = argv[i + 1];
}


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


#ifndef STATICIP

void
getIP(std::string* ip) {
    int fd;
    struct ifreq ifr;

    fd = socket(AF_INET, SOCK_DGRAM, 0);

    ifr.ifr_addr.sa_family = AF_INET;   // Type of address to retrieve - IPv4 IP address.

    strncpy(ifr.ifr_name, INTERFACE, IFNAMSIZ - 1);   // Copy the interface name in the ifreq structure.

    ioctl(fd, SIOCGIFADDR, &ifr);

    close(fd);

    *ip = inet_ntoa(((struct sockaddr_in *) &ifr.ifr_addr)->sin_addr);
    fprintf(stderr, "IP Address: %s\n", ip->c_str());

    return;
}

#else

void
getIP(std::string* ip) {
    std::ifstream inFile("/home/keval/Desktop/workspace/aspire/run/tmp/myip");
    inFile >> *ip;
    fprintf(stderr, "IP Address: %s\n", ip->c_str());
}

#endif

