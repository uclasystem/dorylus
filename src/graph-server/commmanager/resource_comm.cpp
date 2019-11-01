#include "resource_comm.hpp"
#include <iostream>

ResourceComm *createResourceComm(const std::string &type, CommInfo &commInfo) {
    void *hndl = NULL;
    if (type == "GPU")
        hndl = dlopen("./build/graph-server/commmanager/libgpu_comm.so", RTLD_NOW);
    if (type == "Lambda")
        hndl = dlopen("./build/graph-server/commmanager/liblambda_comm.so", RTLD_NOW);

    if (hndl == NULL) {
        std::cerr << dlerror() << std::endl;
        exit(-1);
    }
    void *creator = dlsym(hndl, "createComm");
    ResourceComm *resComm = ((ResourceComm * (*)(CommInfo &)) creator)(commInfo);

    return resComm;
}

void destroyResourceComm(const std::string &type, ResourceComm *resComm) {
    void *hndl = NULL;
    if (type == "GPU")
        hndl = dlopen("./build/graph-server/commmanager/libgpu_comm.so", RTLD_NOW);
    if (type == "Lambda")
        hndl = dlopen("./build/graph-server/commmanager/liblambda_comm.so", RTLD_NOW);

    if (hndl == NULL) {
        std::cerr << dlerror() << std::endl;
        exit(-1);
    }
    void *destroyer = dlsym(hndl, "destroyComm");
    ((void (*)(ResourceComm *)) destroyer)(resComm);
}