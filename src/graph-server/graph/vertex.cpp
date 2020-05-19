#include <iostream>
#include <cassert>
#include <iterator>
#include "vertex.hpp"
#include "graph.hpp"


/////////////////////////
// For `Vertex` class. //
/////////////////////////

unsigned
Vertex::getSourceVertexLocalId(unsigned i) {
    assert(i < inEdges.size());
    if (inEdges[i].getEdgeLocation() == LOCAL_EDGE_TYPE) {
        assert(graph_ptr->localToGlobalId.find(inEdges[i].getSourceId()) != graph_ptr->localToGlobalId.end());
        return inEdges[i].getSourceId();
    } else {
        unsigned gvid = inEdges[i].getSourceId();
        assert(graph_ptr->containsInEdgeGhostVertex(gvid));
        return graph_ptr->getInEdgeGhostVertex(gvid).getLocalId();
    }
}

unsigned
Vertex::getSourceVertexGlobalId(unsigned i) {
    assert(i < inEdges.size());
    if (inEdges[i].getEdgeLocation() == LOCAL_EDGE_TYPE) {
        assert(graph_ptr->localToGlobalId.find(inEdges[i].getSourceId()) != graph_ptr->localToGlobalId.end());
        return graph_ptr->localToGlobalId[inEdges[i].getSourceId()];
    } else
        return inEdges[i].getSourceId();
}

unsigned
Vertex::getDestVertexLocalId(unsigned i) {
    assert(i < outEdges.size());
    if (outEdges[i].getEdgeLocation() == LOCAL_EDGE_TYPE) {
        assert(graph_ptr->localToGlobalId.find(outEdges[i].getDestId()) != graph_ptr->localToGlobalId.end());
        return outEdges[i].getDestId();
    } else {
        unsigned gvid = outEdges[i].getDestId();
        assert(graph_ptr->containsOutEdgeGhostVertex(gvid));
        return graph_ptr->getOutEdgeGhostVertex(gvid).getLocalId();
    }
}

unsigned
Vertex::getDestVertexGlobalId(unsigned i) {
    assert(i < outEdges.size());
    if (outEdges[i].getEdgeLocation() == LOCAL_EDGE_TYPE) {
        assert(graph_ptr->localToGlobalId.find(outEdges[i].getDestId()) != graph_ptr->localToGlobalId.end());
        return graph_ptr->localToGlobalId[outEdges[i].getDestId()];
    } else
        return outEdges[i].getDestId();
}

void
Vertex::compactVertex() {
    inEdges.shrink_to_fit();
    outEdges.shrink_to_fit();
}


//////////////////////////////
// For `GhostVertex` class. //
//////////////////////////////

void
GhostVertex::compactVertex() {
    edges.shrink_to_fit();
}
