#ifndef __VERTEX_HPP__
#define __VERTEX_HPP__

#include <atomic>
#include <vector>
#include <pthread.h>
#include "../parallel/rwlock.hpp"
#include "edge.cpp"

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

        //VertexType oldVertexData;

        Graph<VertexType, EdgeType>* graph;

        Vertex();
        ~Vertex();
        IdType localId();
        IdType globalId();

        VertexType data();                      // Get the current value.
        VertexType dataAt(unsigned layer);      // Get value at specific layer.
        std::vector<VertexType>& dataAll();     // Get reference to all old values' vector.
        void setData(VertexType value);         // Modify the current value.
        void addData(VertexType value);         // Add a new value of the new iteration.

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
        VertexType getSourceVertexDataAt(unsigned i, unsigned layer);
        VertexType getDestVertexData(unsigned i);
        unsigned getSourceVertexGlobalId(unsigned i);
        unsigned getDestVertexGlobalId(unsigned i);

        IdType parent();
        void setParent(IdType p); 
};


template <typename VertexType>
class GhostVertex {
    public:
        
        // Use a vector to make data in old iterations persistent.
        std::vector<VertexType> vertexData;

        //char version;
        RWLock lock;
        int32_t degree;

        std::vector<IdType> outEdges;

        GhostVertex() {
            lock.init();
            degree = 0;
        }

        GhostVertex(const VertexType vData) {
            lock.init();
            degree = 0;
            vertexData.clear();
            vertexData.push_back(vData);
        }

        ~GhostVertex() {
            lock.destroy();
        }

        VertexType data() {     // Get the current value.
            lock.readLock();
            VertexType vData = vertexData.back();
            lock.unlock();
            return vData;
        }

        void setData(VertexType* value) {   // Modify the current value.
            lock.writeLock();
            vertexData.back() = *value;
            lock.unlock();
        }

        void addData(VertexType* value) {   // Add a new value of the new iteration.
            lock.writeLock();
            vertexData.push_back(*value);
            lock.unlock();
        }

        void incrementDegree() { ++degree; }
};


#endif /* __VERTEX_HPP__ */
