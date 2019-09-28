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
    Vertex& getVertex(unsigned lvid);
    Vertex& getVertexByGlobal(unsigned gvid);
    bool containsVertex(unsigned gvid);   // Contain searches with global ID.

    std::map<unsigned, GhostVertex>& getInEdgeGhostVertices()  { return inEdgeGhostVertices; }
    std::map<unsigned, GhostVertex>& getOutEdgeGhostVertices() { return outEdgeGhostVertices; }
    GhostVertex& getInEdgeGhostVertex(unsigned gvid);
    GhostVertex& getOutEdgeGhostVertex(unsigned gvid);
    bool containsInEdgeGhostVertex(unsigned gvid);
    bool containsOutEdgeGhostVertex(unsigned gvid);

    unsigned getNumLocalVertices() { return numLocalVertices; }
    void setNumLocalVertices(unsigned num) { numLocalVertices = num; }
    unsigned getNumGlobalVertices() { return numGlobalVertices; }
    void setNumGlobalVertices(unsigned num) { numGlobalVertices = num; }
    unsigned getNumInEdgeGhostVertices()  { return numInEdgeGhostVertices; }
    unsigned getNumOutEdgeGhostVertices() { return numOutEdgeGhostVertices; }
    void setNumInEdgeGhostVertices(unsigned num)  { numInEdgeGhostVertices = num; }
    void setNumOutEdgeGhostVertices(unsigned num) { numOutEdgeGhostVertices = num; }

    unsigned long long getNumGlobalEdges() { return numGlobalEdges; }
    void incrementNumGlobalEdges() { ++numGlobalEdges; }

    short getVertexPartitionId(unsigned vid) { return vertexPartitionIds[vid]; }
    void appendVertexPartitionId(short pid) { vertexPartitionIds.push_back(pid); }

    void compactGraph();

    std::map<unsigned, unsigned> globalToLocalId;
    std::map<unsigned, unsigned> localToGlobalId;

private:

    std::vector<Vertex> vertices;
    std::map<unsigned, GhostVertex> inEdgeGhostVertices;
    std::map<unsigned, GhostVertex> outEdgeGhostVertices;

    unsigned numLocalVertices;
    unsigned numGlobalVertices;
    unsigned numInEdgeGhostVertices;
    unsigned numOutEdgeGhostVertices;

    unsigned long long numGlobalEdges = 0;

    std::vector<short> vertexPartitionIds;
};


#endif //__GRAPH_HPP__
