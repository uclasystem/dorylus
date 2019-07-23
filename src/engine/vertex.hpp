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
    InEdge<EdgeType>& getInEdge(unsigned i);
    void addInEdge(InEdge<EdgeType> edge);
    OutEdge<EdgeType>& getOutEdge(unsigned i);
    void addOutEdge(OutEdge<EdgeType> edge);

    VertexType getSourceVertexData(unsigned i);
    VertexType getSourceVertexDataAt(unsigned i, unsigned layer);
    unsigned getSourceVertexGlobalId(unsigned i);
    unsigned getDestVertexGlobalId(unsigned i);

    Graph<VertexType, EdgeType> *getGraphPtr();
    void setGraphPtr(Graph<VertexType, EdgeType> *ptr);

    IdType getParent();
    void setParent(IdType p);

    void compactVertex();

private:

    IdType localId;
    IdType globalId;

    std::vector<VertexType> vertexData;     // Use a vector to make data in old iterations persistent.
    VertexLocationType vertexLocation;

    std::vector< InEdge<EdgeType> > inEdges;
    std::vector< OutEdge<EdgeType> > outEdges;

    IdType parentId;

    Graph<VertexType, EdgeType> *graph_ptr;

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
