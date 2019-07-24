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


class Graph;


/**
 *
 * Class for a local vertex.
 * 
 */
class Vertex {

public:

    Vertex() : localId(0), globalId(0), parentId(MAX_IDTYPE), graph_ptr(NULL) { lock.init(); }
    ~Vertex() { lock.destroy(); }

    IdType getLocalId() { return localId; }
    void setLocalId(IdType lvid) { localId = lvid; }
    IdType getGlobalId() { return globalId; }
    void setGlobalId(IdType gvid) { globalId = gvid; }

    VertexLocationType getVertexLocation() { return vertexLocation; }
    void setVertexLocation(VertexLocationType loc) { vertexLocation = loc; }

    unsigned getNumInEdges() { return inEdges.size(); }
    unsigned getNumOutEdges() { return outEdges.size(); }
    InEdge& getInEdge(unsigned i) { return inEdges[i]; }
    void addInEdge(InEdge edge) { inEdges.push_back(edge); }
    OutEdge& getOutEdge(unsigned i) { return outEdges[i]; }
    void addOutEdge(OutEdge edge) { outEdges.push_back(edge); }

    IdType getSourceVertexLocalId(unsigned i);
    IdType getSourceVertexGlobalId(unsigned i);
    IdType getDestVertexLocalId(unsigned i);
    IdType getDestVertexGlobalId(unsigned i);

    Graph *getGraphPtr() { return graph_ptr; }
    void setGraphPtr(Graph *ptr) { graph_ptr = ptr; }

    IdType getParent() { return parentId; }
    void setParent(IdType p) { parentId = p; }

    void compactVertex();

    void aggregateFromNeighbors();
    void produceOutput();

    void readLock() { lock.readLock(); }
    void writeLock() { lock.writeLock(); }
    void unlock() { lock.unlock(); }

private:

    IdType localId;
    IdType globalId;

    VertexLocationType vertexLocation;

    std::vector<InEdge> inEdges;
    std::vector<OutEdge> outEdges;

    IdType parentId;

    Graph *graph_ptr;

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

    IdType getLocalId() { return localId; }
    void setLocalId(IdType id) { localId = id; }

    void addOutEdge(IdType dId) { outEdges.push_back(dId); }

    int32_t getDegree() { return degree; }
    void incrementDegree() { ++degree; }

    void compactVertex();

    void readLock() { lock.readLock(); }
    void writeLock() { lock.writeLock(); }
    void unlock() { lock.unlock(); }

private:

    IdType localId;     // Added to serve as the index in the static values region.

    std::vector<IdType> outEdges;
    int32_t degree;

    RWLock lock;
};


#endif // __VERTEX_HPP__
