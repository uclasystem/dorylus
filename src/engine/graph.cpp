#ifndef __GRAPH_CPP__
#define __GRAPH_CPP__


#include <cassert>
#include "graph.hpp"


template <typename VertexType, typename EdgeType>
std::vector< Vertex<VertexType, EdgeType> >&
Graph<VertexType, EdgeType>::getVertices() {
    return vertices;
}

template <typename VertexType, typename EdgeType>
Vertex<VertexType, EdgeType>&
Graph<VertexType, EdgeType>::getVertex(IdType lvid) {
    assert(lvid < vertices.size());
    return vertices[lvid];
}

template <typename VertexType, typename EdgeType>
Vertex<VertexType, EdgeType>&
Graph<VertexType, EdgeType>::getVertexByGlobal(IdType gvid) {
    assert(globalToLocalId.find(gvid) != globalToLocalId.end());
    return vertices[globalToLocalId[gvid]];
}

template <typename VertexType, typename EdgeType>
bool
Graph<VertexType, EdgeType>::containsVertex(IdType gvid) {
    return globalToLocalId.find(gvid) != globalToLocalId.end();
}

template <typename VertexType, typename EdgeType>
std::map< IdType, GhostVertex<VertexType> >&
Graph<VertexType, EdgeType>::getGhostVertices() {
    return ghostVertices;
}

template <typename VertexType, typename EdgeType>
GhostVertex<VertexType>&
Graph<VertexType, EdgeType>::getGhostVertex(IdType gvid) {
    assert(ghostVertices.find(gvid) != ghostVertices.end());
    return ghostVertices[gvid];
}

template <typename VertexType, typename EdgeType>
bool
Graph<VertexType, EdgeType>::containsGhostVertex(IdType gvid) {
    return ghostVertices.find(gvid) != ghostVertices.end();
}

template <typename VertexType, typename EdgeType>
IdType
Graph<VertexType, EdgeType>::getNumLocalVertices() {
    return numLocalVertices;
}

template <typename VertexType, typename EdgeType>
void
Graph<VertexType, EdgeType>::setNumLocalVertices(IdType num) {
    numLocalVertices = num;
}

template <typename VertexType, typename EdgeType>
IdType
Graph<VertexType, EdgeType>::getNumGlobalVertices() {
    return numGlobalVertices;
}

template <typename VertexType, typename EdgeType>
void
Graph<VertexType, EdgeType>::setNumGlobalVertices(IdType num) {
    numGlobalVertices = num;
}

template <typename VertexType, typename EdgeType>
unsigned long long
Graph<VertexType, EdgeType>::getNumGlobalEdges() {
    return numGlobalEdges;
}

template <typename VertexType, typename EdgeType>
std::vector<short>&
Graph<VertexType, EdgeType>::getVertexPartitionIds() {
    return vertexPartitionIds;
}


/**
 *
 * Update a ghost vertex's value.
 * 
 */
template <typename VertexType, typename EdgeType>
void
Graph<VertexType, EdgeType>::updateGhostVertex(IdType vId, VertexType value) {
    typename std::map<IdType, GhostVertex<VertexType> >::iterator it = ghostVertices.find(vId);
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
template <typename VertexType, typename EdgeType>
void
Graph<VertexType, EdgeType>::compactGraph() {
    vertexPartitionIds.shrink_to_fit();
    vertices.shrink_to_fit();
    for (IdType i = 0; i < vertices.size(); ++i) {
        vertices[i].getInEdges().shrink_to_fit();
        vertices[i].getOutEdges().shrink_to_fit(); 
    }
    typename std::map<IdType, GhostVertex<VertexType> >::iterator it;
    for (it = ghostVertices.begin(); it != ghostVertices.end(); ++it)
        it->second.getOutEdges().shrink_to_fit(); 
}


#endif // __GRAPH_CPP__
