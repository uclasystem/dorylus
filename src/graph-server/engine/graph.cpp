#include <cassert>
#include "graph.hpp"


Vertex&
Graph::getVertex(unsigned lvid) {
    assert(lvid < vertices.size());
    return vertices[lvid];
}

Vertex&
Graph::getVertexByGlobal(unsigned gvid) {
    assert(globalToLocalId.find(gvid) != globalToLocalId.end());
    return vertices[globalToLocalId[gvid]];
}

bool
Graph::containsVertex(unsigned gvid) {
    return globalToLocalId.find(gvid) != globalToLocalId.end();
}

GhostVertex&
Graph::getInEdgeGhostVertex(unsigned gvid) {
    assert(inEdgeGhostVertices.find(gvid) != inEdgeGhostVertices.end());
    return inEdgeGhostVertices[gvid];
}

GhostVertex&
Graph::getOutEdgeGhostVertex(unsigned gvid) {
    assert(outEdgeGhostVertices.find(gvid) != outEdgeGhostVertices.end());
    return outEdgeGhostVertices[gvid];
}

bool
Graph::containsInEdgeGhostVertex(unsigned gvid) {
    return inEdgeGhostVertices.find(gvid) != inEdgeGhostVertices.end();
}

bool
Graph::containsOutEdgeGhostVertex(unsigned gvid) {
    return outEdgeGhostVertices.find(gvid) != outEdgeGhostVertices.end();
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

    for (unsigned i = 0; i < vertices.size(); ++i)
        vertices[i].compactVertex();

    for (auto it = inEdgeGhostVertices.begin(); it != inEdgeGhostVertices.end(); ++it) {
        it->second.compactVertex();
    }
    for (auto it = outEdgeGhostVertices.begin(); it != outEdgeGhostVertices.end(); ++it) {
        it->second.compactVertex();
    }
}
