#ifndef __GRAPH_CPP__
#define __GRAPH_CPP__


#include <cassert>
#include "graph.hpp"


/**
 *
 * Update a ghost vertex's value.
 * 
 */
template <typename VertexType, typename EdgeType>
void Graph<VertexType, EdgeType>::updateGhostVertex(IdType vId, VertexType value) {
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
void Graph<VertexType, EdgeType>::compactGraph() {
    vertexPartitionIds.shrink_to_fit();
    vertices.shrink_to_fit();
    for (IdType i = 0; i < vertices.size(); ++i) {
        vertices[i].inEdges.shrink_to_fit();
        vertices[i].outEdges.shrink_to_fit(); 
    }
    typename std::map<IdType, GhostVertex<VertexType> >::iterator it;
    for (it = ghostVertices.begin(); it != ghostVertices.end(); ++it)
        it->second.outEdges.shrink_to_fit(); 
}


#endif // __GRAPH_CPP__
