#ifndef __GRAPH_HPP__
#define __GRAPH_HPP__

#include "graph.hpp"

template <typename VertexType, typename EdgeType>
VertexType Graph<VertexType, EdgeType>::getVertexValue(IdType vId) {
//1. look at where it belongs
//2. send appropriate value, else assert<false>
    assert(false);
    return VertexType();
}

template <typename VertexType, typename EdgeType>
void Graph<VertexType, EdgeType>::updateGhostVertex(IdType vId, VertexType* value) {
    typename std::map<IdType, GhostVertex<VertexType> >::iterator it = ghostVertices.find(vId);
    if(it == ghostVertices.end()) {
        fprintf(stderr, "No ghost vertex with id %u\n", vId);
    }
    assert(it != ghostVertices.end());
    it->second.addData(value);
}

template <typename VertexType, typename EdgeType>
void Graph<VertexType, EdgeType>::printGraphMetrics() {
    fprintf(stderr, "Graph Metrics: numGlobalVertices = %u\n", numGlobalVertices);
    fprintf(stderr, "Graph Metrics: numGlobalEdges = %llu\n", numGlobalEdges);
    fprintf(stderr, "Graph Metrics: numLocalVertices = %u\n", numLocalVertices);
}

template <typename VertexType, typename EdgeType>
void Graph<VertexType, EdgeType>::compactGraph() {
    vertexPartitionIds.shrink_to_fit();
    vertices.shrink_to_fit();
    for(IdType i=0; i<vertices.size(); ++i) {
        vertices[i].inEdges.shrink_to_fit();
        vertices[i].outEdges.shrink_to_fit(); 
    }
    typename std::map<IdType, GhostVertex<VertexType> >::iterator it;
    for(it = ghostVertices.begin(); it != ghostVertices.end(); ++it)
        it->second.outEdges.shrink_to_fit(); 

}

template <typename VertexType>
void ThinGraph<VertexType>::compactGraph() {
    vertices.shrink_to_fit();

    typename std::map<IdType, GhostVertex<VertexType> >::iterator it;
    for(it = ghostVertices.begin(); it != ghostVertices.end(); ++it)
        it->second.outEdges.shrink_to_fit();
}

#endif //__GRAPH_HPP__
