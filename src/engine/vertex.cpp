#include <iostream>
#include <cassert>
#include <iterator>
#include "vertex.hpp"
#include "graph.hpp"


/////////////////////////
// For `Vertex` class. //
/////////////////////////

IdType
Vertex::getSourceVertexLocalId(unsigned i) {
    assert(i < inEdges.size());
    if (inEdges[i].getEdgeLocation() == LOCAL_EDGE_TYPE) {
        assert(graph_ptr->localToGlobalId.find(inEdges[i].getSourceId()) != graph_ptr->localToGlobalId.end());
        return inEdges[i].getSourceId();
    } else {
        IdType gvid = inEdges[i].getSourceId();
        assert(graph_ptr->containsGhostVertex(gvid));
        return graph_ptr->getGhostVertex(gvid).getLocalId();
    }
}

IdType
Vertex::getSourceVertexGlobalId(unsigned i) {
    assert(i < inEdges.size());
    if (inEdges[i].getEdgeLocation() == LOCAL_EDGE_TYPE) {
        assert(graph_ptr->localToGlobalId.find(inEdges[i].getSourceId()) != graph_ptr->localToGlobalId.end());
        return graph_ptr->localToGlobalId[inEdges[i].getSourceId()];
    } else
        return inEdges[i].getSourceId();
}

IdType
Vertex::getDestVertexLocalId(unsigned i) {
    assert(i < outEdges.size());
    if (outEdges[i].getEdgeLocation() == LOCAL_EDGE_TYPE) {
        assert(graph_ptr->localToGlobalId.find(outEdges[i].getDestId()()) != graph_ptr->localToGlobalId.end());
        return outEdges[i].getDestId();
    } else {
        IdType gvid = outEdges[i].getDestId();
        assert(graph_ptr->containsGhostVertex(gvid));
        return graph_ptr->getGhostVertex(gvid).getLocalId();
    }
}

IdType
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
    outEdges.shrink_to_fit();
}
