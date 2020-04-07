#ifndef __GRAPH_UTILS_HPP__
#define __GRAPH_UTILS_HPP__


#include <algorithm>
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


/** Default vertex ID type and features type. */
typedef float FeatType;
typedef float EdgeType;
#define MAX_IDTYPE UINT_MAX     // Limit: MAX_IDTYPE must be at least two times the number of global vertices.
                                // From 0 to numGlobalVertices are normal ghost vertices update message,
                                // From MAX_IDTYPE downto MAX_IDTYPE - numGlobalVertices are receive signals.

struct Chunk {
    unsigned chunkId;
    unsigned lowBound;
    unsigned upBound;
    unsigned layer;
    PROP_TYPE dir;

    unsigned epoch;

    bool vertex;

    bool operator<(const Chunk &rhs) const {
        // TODO: (YIFAN) Assign priority in the computation sequence
        return
            epoch < rhs.epoch || (epoch == rhs.epoch && (
            dir < rhs.dir || (dir == rhs.dir && (
            layer < rhs.layer || (layer == rhs.layer && (
            (vertex && !rhs.vertex) || (vertex == rhs.vertex && (
            chunkId < rhs.chunkId || (chunkId == rhs.chunkId && (
            lowBound < rhs.lowBound || (lowBound == rhs.lowBound && (
            upBound < rhs.upBound))))))))))));
    }
};

Chunk createChunk(unsigned rows, unsigned nChunks, unsigned id, unsigned layer, PROP_TYPE dir, unsigned ep = 0, bool vertex = true);

extern std::map<size_t, std::string> typeToFormatSpecifier;

typedef std::function<void(unsigned, unsigned, FeatType*, unsigned)> FuncPtr;
typedef std::queue< Chunk > ChunkQueue;
typedef std::queue< std::pair<unsigned, unsigned> > PairQueue;


/** Print to log file using this one. */
void printLog(const unsigned nodeId, const char *format, ...);


/** Acquire timer value using this one. */
double getTimer();
std::time_t getCurrentTime();


void getPrIP(std::string& myPrIpFile, std::string& ip);


inline size_t argmax(FeatType* first, FeatType* last) { return std::distance(first, std::max_element(first, last)); }

#endif //__GRAPH_UTILS_HPP__
