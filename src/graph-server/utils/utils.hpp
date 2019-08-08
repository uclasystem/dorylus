#ifndef __GRAPH_UTILS_HPP__
#define __GRAPH_UTILS_HPP__


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
#define MAX_IDTYPE UINT_MAX     // Limit: MAX_IDTYPE must be at least two times the number of global vertices.
                                // From 0 to numGlobalVertices are normal ghost vertices update message,
                                // From MAX_IDTYPE downto MAX_IDTYPE - numGlobalVertices are receive signals.


#define INTERFACE "ens5" 


extern std::map<size_t, std::string> typeToFormatSpecifier;


/** Print to log file using this one. */
void printLog(const unsigned nodeId, const char *format, ...);


/** Acquire timer value using this one. */
double getTimer();


void getPrIP(std::string& myPrIpFile, std::string& ip);
void getPubIP(std::string& myPubIpFile, std::string& ip);


#endif //__GRAPH_UTILS_HPP__
