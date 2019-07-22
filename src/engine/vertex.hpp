#ifndef __VERTEX_HPP__
#define __VERTEX_HPP__


#include <atomic>
#include <vector>
#include <pthread.h>
#include "../parallel/rwlock.hpp"
#include "edge.cpp"


/** Vertex type indicators. */
typedef char VertexLocationType;
#define INTERNAL_VERTEX 'I'
#define BOUNDARY_VERTEX 'B'


template<typename VertexType, typename EdgeType>
class Graph;


/**
 *
 * Class for a local vertex.
 * 
 */
template<typename VertexType, typename EdgeType>
class Vertex {

public:

    Vertex();
    ~Vertex();

    IdType localId();
    IdType globalId();

    VertexType data();                      // Get the current value.
    VertexType dataAt(unsigned layer);      // Get value at specified layer.
    std::vector<VertexType>& dataAll();     // Get reference to all old values' vector.

    void setData(VertexType value);         // Modify the current value.
    void addData(VertexType value);         // Add a new value of the new iteration.

    unsigned numInEdges();
    unsigned numOutEdges();

    InEdge<EdgeType>& getInEdge(unsigned i);
    OutEdge<EdgeType>& getOutEdge(unsigned i);

    VertexType getSourceVertexData(unsigned i);
    VertexType getSourceVertexDataAt(unsigned i, unsigned layer);

    unsigned getSourceVertexGlobalId(unsigned i);
    unsigned getDestVertexGlobalId(unsigned i);

    IdType parent();
    void setParent(IdType p);

private:

    IdType localIdx;
    IdType globalIdx;

    std::vector<VertexType> vertexData;     // Use a vector to make data in old iterations persistent.
    VertexLocationType vertexLocation;

    std::vector< InEdge<EdgeType> > inEdges;
    std::vector< OutEdge<EdgeType> > outEdges;

    Graph<VertexType, EdgeType> *graph;

    IdType parentIdx;

    RWLock lock;
};


/**
 *
 * Class for a ghost vertex.
 * 
 */
template <typename VertexType>
class GhostVertex {

public:
    
    GhostVertex(const VertexType vData);
    ~GhostVertex();

    VertexType data();                      // Get the current value.
    VertexType dataAt(unsigned layer);      // Get value at specified layer.

    void setData(VertexType value);         // Modify the current value.
    void addData(VertexType value);         // Add a new value of the new iteration.

    void incrementDegree() {
        ++degree;
    }

private:

    std::vector<VertexType> vertexData;

    std::vector<IdType> outEdges;
    int32_t degree;

    RWLock lock;
};


#endif // __VERTEX_HPP__
