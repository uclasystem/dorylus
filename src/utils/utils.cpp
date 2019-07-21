#include <unistd.h>
#include <sys/time.h>
#include <fstream>
#include <iostream>
#include <sys/ioctl.h>
#include <net/if.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <cstring>
#include "utils.hpp"

std::map<size_t, std::string> typeToFormatSpecifier;

void init() {
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

double getTimer() {
    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec * 1000 + t.tv_usec / 1000.0;
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


#ifndef STATICIP
void getIP(std::string* ip) {
    int fd;
    struct ifreq ifr;

    fd = socket(AF_INET, SOCK_DGRAM, 0);

    //Type of address to retrieve - IPv4 IP address
    ifr.ifr_addr.sa_family = AF_INET;

    //Copy the interface name in the ifreq structure
    strncpy(ifr.ifr_name, INTERFACE, IFNAMSIZ-1);

    ioctl(fd, SIOCGIFADDR, &ifr);

    close(fd);

    //display result
    //printf("%s - %s\n" , iface , inet_ntoa(((struct sockaddr_in*) &ifr.ifr_addr)->sin_addr));
    *ip = inet_ntoa(((struct sockaddr_in*) &ifr.ifr_addr)->sin_addr);
    fprintf(stderr, "IP Address: %s\n", ip->c_str());

    return;
}

#else
void getIP(std::string* ip) {
    std::ifstream inFile("/home/keval/Desktop/workspace/aspire/run/tmp/myip");
    inFile >> *ip;
    fprintf(stderr, "IP Address: %s\n", ip->c_str());
}
#endif

