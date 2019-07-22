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

    std::vector< Vertex<VertexType, EdgeType> >& getVertices();
    std::map< IdType, GhostVertex<VertexType> >& getGhostVertices();

    IdType getNumLocalVertices();
    IdType getNumGlobalVertices();

    unsigned long long getNumGlobalEdges();

    std::vector<short> getVertexPartitionIds();

    void updateGhostVertex(IdType vId, VertexType value);

    void compactGraph();

    std::map<IdType, IdType> globalToLocalId;
    std::map<IdType, IdType> localToGlobalId;

private:

    std::vector< Vertex<VertexType, EdgeType> > vertices;
    std::map< IdType, GhostVertex<VertexType> > ghostVertices;

    IdType numLocalVertices;
    IdType numGlobalVertices;
    
    unsigned long long numGlobalEdges;

    std::vector<short> vertexPartitionIds;
};


#endif //__GRAPH_HPP__
