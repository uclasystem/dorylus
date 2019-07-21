#ifndef __GRAPH_H__
#define __GRAPH_H__

#include <vector>
#include <map>
#include "../parallel/lock.hpp"
#include "vertex.hpp"
#include "edge.hpp"

/*
struct ScatterVersionType {
    char version;
    Lock lock;

    ScatterVersionType(char ver = 0) : version(ver) {
        lock.init();
    } 

    ~ScatterVersionType() {
        lock.destroy();
    }
};
*/

template <typename VertexType, typename EdgeType>
class Graph {
public:
    //Vertex<VertexType, EdgeType>* vertices;
    std::vector<Vertex<VertexType, EdgeType> > vertices;
    //IdType vIdOffset;
    IdType numLocalVertices;
    IdType numGlobalVertices;
    unsigned long long numGlobalEdges;

    std::vector<short> vertexPartitionIds;

    std::map<IdType, IdType> globalToLocalId;   // TODO: Can this be optimized to have only boundary vertices?
    std::map<IdType, IdType> localToGlobalId;
    
    std::map<IdType, GhostVertex<VertexType> > ghostVertices;
    //std::map<IdType, ScatterVersionType> scatterVersions;

    VertexType getVertexValue(IdType vId);
    void updateGhostVertex(IdType vId, VertexType* value);
    void printGraphMetrics();
    void compactGraph();
};

template <typename VertexType>
class ThinGraph {
public:
    std::vector<VertexType> vertices;
    std::map<IdType, GhostVertex<VertexType> > ghostVertices;
    void compactGraph();
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

        VertexType dataAt(unsigned layer) {     // Get value at specified layer.
            lock.readLock();
            assert(layer < vertexData.size());
            VertexType vData = vertexData[layer];
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


#endif //__GRAPH_H__
