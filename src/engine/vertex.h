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

        // Use a vector to make data in old iterations persistent.
        std::vector<VertexType> vertexData;

        // Use an iteration counter per vertex to track its iteration status.
        int curr_layer = 0;

        //VertexType oldVertexData;

        Graph<VertexType, EdgeType>* graph;

        Vertex();
        ~Vertex();
        IdType localId();
        IdType globalId();

        VertexType data();                      // Get the current value.
        std::vector<VertexType>& dataAll();     // Get reference to all old values' vector.
        void setData(VertexType value);         // Modify the current value.
        void addData(VertexType value);         // Add a new value of the new iteration.

        bool nextLayer();    // Increment the layer counter. Returns whether threshold met (not met -> to continue).

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
