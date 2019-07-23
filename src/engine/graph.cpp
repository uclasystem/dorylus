#ifndef __GRAPH_CPP__
#define __GRAPH_CPP__


#include <cassert>
#include "graph.hpp"


std::vector<Vertex>&
Graph::getVertices() {
    return vertices;
}

Vertex&
Graph::getVertex(IdType lvid) {
    assert(lvid < vertices.size());
    return vertices[lvid];
}

Vertex&
Graph::getVertexByGlobal(IdType gvid) {
    assert(globalToLocalId.find(gvid) != globalToLocalId.end());
    return vertices[globalToLocalId[gvid]];
}

bool
Graph::containsVertex(IdType gvid) {
    return globalToLocalId.find(gvid) != globalToLocalId.end();
}

std::map<IdType, GhostVertex>&
Graph::getGhostVertices() {
    return ghostVertices;
}

GhostVertex&
Graph::getGhostVertex(IdType gvid) {
    assert(ghostVertices.find(gvid) != ghostVertices.end());
    return ghostVertices[gvid];
}

bool
Graph::containsGhostVertex(IdType gvid) {
    return ghostVertices.find(gvid) != ghostVertices.end();
}

IdType
Graph::getNumLocalVertices() {
    return numLocalVertices;
}

void
Graph::setNumLocalVertices(IdType num) {
    numLocalVertices = num;
}

IdType
Graph::getNumGlobalVertices() {
    return numGlobalVertices;
}

void
Graph::setNumGlobalVertices(IdType num) {
    numGlobalVertices = num;
}

unsigned long long
Graph::getNumGlobalEdges() {
    return numGlobalEdges;
}

short
Graph::getVertexPartitionId(IdType vid) {
    return vertexPartitionIds[vid];
}

void
Graph::appendVertexPartitionId(short pid) {
    vertexPartitionIds.push_back(pid);
}


/**
 *
 * Update a ghost vertex's value.
 * 
 */
void
Graph::updateGhostVertex(IdType vId, VertexType value) {
    typename std::map<IdType, GhostVertex>::iterator it = ghostVertices.find(vId);
    assert(it != ghostVertices.end());
    it->second.addData(value);
}


/**
 *
 * Compact a graph's data. Mainly does the following things:
 *     1. Shrink all the vertices' edges vector.
 *     2. Shrink all the ghost vertices' edges vector.
 *     3. Shrink the vertices and partitions vector.
 * 
 */
void
Graph::compactGraph() {
    vertexPartitionIds.shrink_to_fit();
    vertices.shrink_to_fit();

    for (IdType i = 0; i < vertices.size(); ++i)
        vertices[i].compactVertex();

    typename std::map<IdType, GhostVertex>::iterator it;
    for (it = ghostVertices.begin(); it != ghostVertices.end(); ++it)
        it->second.compactVertex(); 
}


#endif // __GRAPH_CPP__
