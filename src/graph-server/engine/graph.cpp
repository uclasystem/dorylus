#include <cassert>
#include "graph.hpp"


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

GhostVertex&
Graph::getGhostVertex(IdType gvid) {
    assert(ghostVertices.find(gvid) != ghostVertices.end());
    return ghostVertices[gvid];
}

bool
Graph::containsGhostVertex(IdType gvid) {
    return ghostVertices.find(gvid) != ghostVertices.end();
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
