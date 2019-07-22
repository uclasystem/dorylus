#ifndef __GRAPH_HPP__
#define __GRAPH_HPP__


#include <vector>
#include <map>
#include "../parallel/lock.hpp"
#include "../utils/utils.hpp"
#include "vertex.cpp"
#include "edge.cpp"


/**
 *
 * Class of a graph, composed of vertices and directed edges.
 * 
 */
template <typename VertexType, typename EdgeType>
class Graph {

public:

    void updateGhostVertex(IdType vId, VertexType value);
    void printGraphMetrics();
    void compactGraph();

private:

    std::vector< Vertex<VertexType, EdgeType> > vertices;
    std::map<IdType, GhostVertex<VertexType> > ghostVertices;

    IdType numLocalVertices;
    IdType numGlobalVertices;
    
    unsigned long long numGlobalEdges;

    std::vector<short> vertexPartitionIds;

    std::map<IdType, IdType> globalToLocalId;
    std::map<IdType, IdType> localToGlobalId;
};


#endif //__GRAPH_HPP__
