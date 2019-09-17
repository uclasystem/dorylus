#include "resource_comm.hpp"
#include <iostream>

ResourceComm* createResourceComm(const std::string& type, CommInfo& commInfo){
	void *hndl=NULL; 
	system("pwd");
	if(type=="GPU")
		hndl= dlopen("", RTLD_NOW);
	if(type=="Lambda")
		hndl= dlopen("", RTLD_NOW);

	if(hndl == NULL){
	   std::cerr << dlerror() << std::endl;
	   exit(-1);
	}
	void * creator= dlsym(hndl, "createComm");
	ResourceComm *resComm = ((ResourceComm* (*)()) creator)();

	return resComm;
}