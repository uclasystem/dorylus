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


#endif //__GRAPH_H__
