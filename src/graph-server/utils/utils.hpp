#ifndef __GRAPH_UTILS_HPP__
#define __GRAPH_UTILS_HPP__


#include <chrono>
#include <climits>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <functional>
#include <map>
#include <string>
#include <typeinfo>
#include <vector>
#include <queue>
#include <utility>

#include "../../common/utils.hpp"


#define MAX_IDTYPE UINT_MAX     // Limit: MAX_IDTYPE must be at least two times the number of global vertices.
                                // From 0 to numGlobalVertices are normal ghost vertices update message,
                                // From MAX_IDTYPE downto MAX_IDTYPE - numGlobalVertices are receive signals.

typedef std::priority_queue< Chunk > ChunkQueue;


/** Print to log file using this one. */
void printLog(const unsigned nodeId, const char *format, ...);


/** Acquire timer value using this one. */
double getTimer();
std::time_t getCurrentTime();


void getPrIP(std::string& myPrIpFile, std::string& ip);

#endif //__GRAPH_UTILS_HPP__
