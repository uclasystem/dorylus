#ifndef __VERTEX_HPP__
#define __VERTEX_HPP__


#include <atomic>
#include <vector>
#include <pthread.h>
#include "../parallel/rwlock.hpp"
#include "edge.hpp"


/** Vertex type indicators. */
typedef char VertexLocationType;
#define INTERNAL_VERTEX 'I'
#define BOUNDARY_VERTEX 'B'


class RawGraph;


/**
 *
 * Class for a local vertex.
 *
 */
class Vertex {

public:

    Vertex() : localId(0), globalId(0), parentId(MAX_IDTYPE), graph_ptr(NULL) { lock.init(); }
    ~Vertex() { lock.destroy(); }

    unsigned getLocalId() { return localId; }
    void setLocalId(unsigned lvid) { localId = lvid; }
    unsigned getGlobalId() { return globalId; }
    void setGlobalId(unsigned gvid) { globalId = gvid; }

    VertexLocationType getVertexLocation() { return vertexLocation; }
    void setVertexLocation(VertexLocationType loc) { vertexLocation = loc; }

    unsigned getNumInEdges() { return inEdges.size(); }
    unsigned getNumOutEdges() { return outEdges.size(); }
    InEdge& getInEdge(unsigned i) { return inEdges[i]; }
    void addInEdge(InEdge edge) { inEdges.push_back(edge); }
    OutEdge& getOutEdge(unsigned i) { return outEdges[i]; }
    void addOutEdge(OutEdge edge) { outEdges.push_back(edge); }

    unsigned getSourceVertexLocalId(unsigned i);
    unsigned getSourceVertexGlobalId(unsigned i);
    unsigned getDestVertexLocalId(unsigned i);
    unsigned getDestVertexGlobalId(unsigned i);

    EdgeType getNormFactor() { return normFactor; }
    void setNormFactor(EdgeType factor) { normFactor = factor; }

    RawGraph *getGraphPtr() { return graph_ptr; }
    void setGraphPtr(RawGraph *ptr) { graph_ptr = ptr; }

    unsigned getParent() { return parentId; }
    void setParent(unsigned p) { parentId = p; }

    void compactVertex();

    void aggregateFromNeighbors();
    void produceOutput();

    void readLock() { lock.readLock(); }
    void writeLock() { lock.writeLock(); }
    void unlock() { lock.unlock(); }

private:

    unsigned localId;
    unsigned globalId;

    VertexLocationType vertexLocation;

    std::vector<InEdge> inEdges;
    std::vector<OutEdge> outEdges;

    EdgeType normFactor;

    unsigned parentId;

    RawGraph *graph_ptr;

    RWLock lock;
};


/**
 *
 * Class for a ghost vertex.
 *
 */
class GhostVertex {

public:

    GhostVertex() : degree(0) { lock.init(); }
    ~GhostVertex() { lock.destroy(); }

    unsigned getLocalId() { return localId; }
    void setLocalId(unsigned id) { localId = id; }

    void addAssocEdge(unsigned dId) { edges.push_back(dId); }

    unsigned getDegree() { return degree; }
    void incrementDegree() { ++degree; }

    void compactVertex();

    void readLock() { lock.readLock(); }
    void writeLock() { lock.writeLock(); }
    void unlock() { lock.unlock(); }

private:

    unsigned localId;     // Added to serve as the index in the static values region.

    std::vector<unsigned> edges;
    unsigned degree;

    RWLock lock;
};


#endif // __VERTEX_HPP__
