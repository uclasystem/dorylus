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
    Vertex<VertexType, EdgeType>& getVertex(IdType lvid);
    Vertex<VertexType, EdgeType>& getVertexByGlobal(IdType gvid);
    bool containsVertex(IdType gvid);   // Contain searches with global ID.

    std::map< IdType, GhostVertex<VertexType> >& getGhostVertices();
    GhostVertex<VertexType>& getGhostVertex(IdType gvid);
    bool containsGhostVertex(IdType gvid);

    IdType getNumLocalVertices();
    void setNumLocalVertices(IdType num);
    IdType getNumGlobalVertices();
    void setNumGlobalVertices(IdType num);

    unsigned long long getNumGlobalEdges();
    void incrementNumGlobalEdges() {
        ++numGlobalEdges;
    }

    std::vector<short>& getVertexPartitionIds();

    void updateGhostVertex(IdType vId, VertexType value);

    void compactGraph();

    std::map<IdType, IdType> globalToLocalId;
    std::map<IdType, IdType> localToGlobalId;

private:

    std::vector< Vertex<VertexType, EdgeType> > vertices;
    std::map< IdType, GhostVertex<VertexType> > ghostVertices;

    IdType numLocalVertices;
    IdType numGlobalVertices;

    unsigned long long numGlobalEdges = 0;

    std::vector<short> vertexPartitionIds;
};


#endif //__GRAPH_HPP__
