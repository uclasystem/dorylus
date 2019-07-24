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

    std::vector<Vertex>& getVertices() { return vertices; }
    Vertex& getVertex(IdType lvid);
    Vertex& getVertexByGlobal(IdType gvid);
    bool containsVertex(IdType gvid);   // Contain searches with global ID.

    std::map<IdType, GhostVertex>& getGhostVertices() { return ghostVertices; }
    GhostVertex& getGhostVertex(IdType gvid);
    bool containsGhostVertex(IdType gvid);

    IdType getNumLocalVertices() { return numLocalVertices; }
    void setNumLocalVertices(IdType num) { numLocalVertices = num; }
    IdType getNumGlobalVertices() { return numGlobalVertices; }
    void setNumGlobalVertices(IdType num) { numGlobalVertices = num; }
    IdType getNumGhostVertices() { return numGhostVertices; }
    void setNumGhostVertices(IdType num) { numGhostVertices = num; }

    unsigned long long getNumGlobalEdges() { return numGlobalEdges; }
    void incrementNumGlobalEdges() { ++numGlobalEdges; }

    short getVertexPartitionId(IdType vid) { return vertexPartitionIds[vid]; }
    void appendVertexPartitionId(short pid) { vertexPartitionIds.push_back(pid); }

    void compactGraph();

    std::map<IdType, IdType> globalToLocalId;
    std::map<IdType, IdType> localToGlobalId;

private:

    std::vector<Vertex> vertices;
    std::map<IdType, GhostVertex> ghostVertices;

    IdType numLocalVertices;
    IdType numGlobalVertices;
    IdType numGhostVertices;

    unsigned long long numGlobalEdges = 0;

    std::vector<short> vertexPartitionIds;
};


#endif //__GRAPH_HPP__
