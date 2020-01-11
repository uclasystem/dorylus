#include "resource_comm.hpp"
#include <iostream>


ResourceComm *createResourceComm(const unsigned &type, CommInfo &commInfo) {
    void *hndl = NULL;
    if (type == CPU)
        hndl = dlopen("./build/graph-server/commmanager/libcpu_comm.so", RTLD_NOW);
    if (type == GPU)
        hndl = dlopen("./build/graph-server/commmanager/libgpu_comm.so", RTLD_NOW);
    if (type == LAMBDA)
        hndl = dlopen("./build/graph-server/commmanager/liblambda_comm.so", RTLD_NOW);

    if (hndl == NULL) {
        std::cerr << dlerror() << std::endl;
        exit(-1);
    }
    void *creator = dlsym(hndl, "createComm");
    ResourceComm *resComm = ((ResourceComm * (*)(CommInfo &)) creator)(commInfo);

    return resComm;
}

void destroyResourceComm(const unsigned &type, ResourceComm *resComm) {
    void *hndl = NULL;
    if (type == CPU)
        hndl = dlopen("./build/graph-server/commmanager/libcpu_comm.so", RTLD_NOW);
    if (type == GPU)
        hndl = dlopen("./build/graph-server/commmanager/libgpu_comm.so", RTLD_NOW);
    if (type == LAMBDA)
        hndl = dlopen("./build/graph-server/commmanager/liblambda_comm.so", RTLD_NOW);

    if (hndl == NULL) {
        std::cerr << dlerror() << std::endl;
        exit(-1);
    }
    void *destroyer = dlsym(hndl, "destroyComm");
    ((void (*)(ResourceComm *)) destroyer)(resComm);
}
