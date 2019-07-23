#ifndef __UTILS_HPP__
#define __UTILS_HPP__


#include <cstdio>
#include <map>
#include <cstring>
#include <string>
#include <typeinfo>
#include <climits>
#include <cstdarg>
#include <vector>


/** Default vertex ID type and features type. */
typedef unsigned IdType;
typedef float FeatType;
typedef float EdgeType;
typedef vector<FeatType> VertexType;
#define MAX_IDTYPE UINT_MAX


#define INTERFACE "eth0" 


#ifdef VERBOSE_ERRORS
#define efprintf(ofile, fmt, ...) fprintf(ofile, "%s:%s():%d || " fmt, __FILE__, __func__, __LINE__, __VA_ARGS__);
#else
#define efprintf(ofile, fmt, ...) fprintf(ofile, fmt, __VA_ARGS__);
#endif


extern std::map<size_t, std::string> typeToFormatSpecifier;


/** Global helper functions for command line arguments. */
void initArgs();
void removeArg(int *argc, char **argv, int i);

template<typename ParseType>
bool parseArgs(int *argc, char **argv, const char *str, ParseType *value) {
    int siz = strlen(str);
    for (int i = 1; i < *argc; i++)
        if (strncmp(argv[i], str, siz) == 0) {
            sscanf(argv[i] + siz, typeToFormatSpecifier[typeid(ParseType).hash_code()].c_str(), value);
            removeArg(argc, argv, i);
            return true;
        }
    return false;
}


/** Print to log file using this one. */
void printLog(const unsigned nodeId, const char *format, ...);


/** Acquire timer value using this one. */
double getTimer();


void getIP(std::string *ip);


#endif //__UTILS_HPP__
