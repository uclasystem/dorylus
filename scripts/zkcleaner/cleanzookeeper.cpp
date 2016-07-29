#include "../../src/nodemanager/zkinterface.h"
#include <iostream>
#include <string>
#include <fstream>

#define ZKHOST_FILE "../../config/zkhostfile"

std::string parseZooConfig(const char* zooHostFile) {
    std::string zooHostPort;
    bool first = true;
    std::ifstream inFile(zooHostFile);
    std::string host, port;
    while (inFile >> host >> port) {
        if(first == false)
            zooHostPort += ",";
        zooHostPort += host + ":" + port;
        first = false;
    }
    return zooHostPort;
}

int main(int argc, char* argv[]) {
    if(argc < 2) {
        fprintf(stderr, "Dude! Invoke this: %s <root-name> [<zoo-host-file>]", argv[0]);
        return -1;
    }

    std::string zooHostFile = argc == 3 ? argv[2] : ZKHOST_FILE;

    std::string zooHostPort = parseZooConfig(zooHostFile.c_str());
    
    fprintf(stderr, "Initing zookeeper connection with %s\n", zooHostPort.c_str());
    ZKInterface::init(zooHostPort.c_str());

    fprintf(stderr, "Recursively deleting node %s\n", argv[1]);
    ZKInterface::recursiveDeleteZKNode(argv[1]);  

    return 0;
}
