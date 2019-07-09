#ifndef __UTILS_H__
#define __UTILS_H__

#include <cstdio>
#include <map>
#include <cstring>
#include <string>
#include <typeinfo>
#include <climits>

#define INTERFACE "eth0" 

#ifdef VERBOSE_ERRORS
#define efprintf(ofile, fmt, ...) fprintf(ofile, "%s:%s():%d || " fmt, __FILE__, __func__, __LINE__, __VA_ARGS__);
#else
#define efprintf(ofile, fmt, ...) fprintf(ofile, fmt, __VA_ARGS__);
#endif

typedef unsigned IdType;
typedef unsigned FeatType;
#define MAX_IDTYPE UINT_MAX

typedef struct empty { } Empty;

extern std::map<size_t, std::string> typeToFormatSpecifier;

void init();
void removeArg(int *argc, char **argv, int i);

double getTimer();

template<typename ParseType>
bool parse(int* argc, char** argv, const char* str, ParseType* value) {
  int siz = strlen(str);
  for (int i = 1; i < *argc; i++)
    if (strncmp(argv[i], str, siz) == 0) {
      sscanf(argv[i] + siz, typeToFormatSpecifier[typeid(ParseType).hash_code()].c_str(), value);
      removeArg(argc, argv, i);
      return true;
    }
  return false;
}

void getIP(std::string* ip);

#endif //__UTILS_H__
