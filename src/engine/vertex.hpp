#ifndef __VERTEX_HPP__
#define __VERTEX_HPP__


#include <atomic>
#include <vector>
#include <pthread.h>
#include "../parallel/rwlock.hpp"
#include "edge.hpp"
#include "graph.hpp"


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

    Vertex();
    ~Vertex();

    IdType getLocalId();
    void setLocalId(IdType lvid);
    IdType getGlobalId();
    void setGlobalId(IdType gvid);

    VertexType data();                      // Get the current value.
    VertexType dataAt(unsigned layer);      // Get value at specified layer.
    std::vector<VertexType>& dataAll();     // Get reference to all old values' vector.

    void setData(VertexType value);         // Modify the current value.
    void addData(VertexType value);         // Add a new value of the new iteration.

    VertexLocationType getVertexLocation();
    void setVertexLocation(VertexLocationType loc);

    unsigned getNumInEdges();
    unsigned getNumOutEdges();
    InEdge& getInEdge(unsigned i);
    void addInEdge(InEdge edge);
    OutEdge& getOutEdge(unsigned i);
    void addOutEdge(OutEdge edge);

    VertexType getSourceVertexData(unsigned i);
    VertexType getSourceVertexDataAt(unsigned i, unsigned layer);
    unsigned getSourceVertexGlobalId(unsigned i);
    unsigned getDestVertexGlobalId(unsigned i);

    Graph *getGraphPtr();
    void setGraphPtr(Graph *ptr);

    IdType getParent();
    void setParent(IdType p);

    void compactVertex();

private:

    IdType localId;
    IdType globalId;

    std::vector<VertexType> vertexData;     // Use a vector to make data in old iterations persistent.
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
    
    GhostVertex();
    ~GhostVertex();

    VertexType data();                      // Get the current value.
    VertexType dataAt(unsigned layer);      // Get value at specified layer.

    void setData(VertexType value);         // Modify the current value.
    void addData(VertexType value);         // Add a new value of the new iteration.

    void addOutEdge(IdType dId);

    int32_t getDegree();
    void incrementDegree() {
        ++degree;
    }

    void compactVertex();

private:

    std::vector<VertexType> vertexData;

    std::vector<IdType> outEdges;
    int32_t degree;

    RWLock lock;
};


#endif // __VERTEX_HPP__
