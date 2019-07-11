#ifndef __VERTEX_H__
#define __VERTEX_H__

#include "../parallel/rwlock.hpp"
#include "edge.hpp"
#include <atomic>
#include <vector>
#include <pthread.h>

#define INTERNAL_VERTEX 'I'
#define BOUNDARY_VERTEX 'B'

typedef char VertexLocationType;

template<typename VertexType, typename EdgeType>
class Graph;

template<typename VertexType, typename EdgeType>
class Vertex {
    public:
        IdType localIdx;
        IdType globalIdx;
        VertexLocationType vertexLocation;
        IdType parentIdx;

        RWLock lock;

        std::vector< InEdge<EdgeType> > inEdges;
        std::vector< OutEdge<EdgeType> > outEdges;

        VertexType vertexData;
        //VertexType oldVertexData;

        Graph<VertexType, EdgeType>* graph;

        Vertex();
        ~Vertex();
        IdType localId();
        IdType globalId();
        VertexType data();
        void setData(VertexType value);

        //VertexType oldData();
        //void setOldData(VertexType value);

        unsigned numInEdges();
        unsigned numOutEdges();
        InEdge<EdgeType>& getInEdge(unsigned i);
        OutEdge<EdgeType>& getOutEdge(unsigned i);
        void readLock();
        void writeLock();
        void unlock();

        EdgeType getInEdgeData(unsigned i);
        EdgeType getOutEdgeData(unsigned i);
        VertexType getSourceVertexData(unsigned i);
        VertexType getDestVertexData(unsigned i);
        unsigned getSourceVertexGlobalId(unsigned i);
        unsigned getDestVertexGlobalId(unsigned i);

        IdType parent();
        void setParent(IdType p); 
};

#endif /* __VERTEX_H__ */
