#ifndef __GRAPH_HPP__
#define __GRAPH_HPP__


#include <vector>
#include <map>
#include "../parallel/lock.hpp"
#include "../utils/utils.hpp"
#include "vertex.hpp"
#include "edge.hpp"


/**
 *
 * Class of a graph, composed of vertices and directed edges.
 * 
 */
class Graph {

public:

    std::vector<Vertex>& getVertices();
    Vertex& getVertex(IdType lvid);
    Vertex& getVertexByGlobal(IdType gvid);
    bool containsVertex(IdType gvid);   // Contain searches with global ID.

    std::map<IdType, GhostVertex>& getGhostVertices();
    GhostVertex& getGhostVertex(IdType gvid);
    bool containsGhostVertex(IdType gvid);

    IdType getNumLocalVertices();
    void setNumLocalVertices(IdType num);
    IdType getNumGlobalVertices();
    void setNumGlobalVertices(IdType num);

    unsigned long long getNumGlobalEdges();
    void incrementNumGlobalEdges() {
        ++numGlobalEdges;
    }

    short getVertexPartitionId(IdType vid);
    void appendVertexPartitionId(short pid);

    void updateGhostVertex(IdType vId, VertexType value);

    void compactGraph();

    std::map<IdType, IdType> globalToLocalId;
    std::map<IdType, IdType> localToGlobalId;

private:

    std::vector<Vertex> vertices;
    std::map<IdType, GhostVertex> ghostVertices;

    IdType numLocalVertices;
    IdType numGlobalVertices;

    unsigned long long numGlobalEdges = 0;

    std::vector<short> vertexPartitionIds;
};


#endif //__GRAPH_HPP__
