#ifndef __EDGE_CPP__
#define __EDGE_CPP__


#include <cstring>
#include "edge.hpp"


////////////////////////////
// For base class `Edge`. //
////////////////////////////

Edge::Edge(IdType oId, EdgeLocationType eLocation, EdgeType eData)
    : otherId(oId), edgeData(eData), edgeLocation(eLocation) { }

EdgeType
Edge::data() {
    return edgeData;
}

EdgeLocationType
Edge::getEdgeLocation() {
    return edgeLocation;
}

void
Edge::setEdgeLocation(EdgeLocationType eLoc) {
    edgeLocation = eLoc;
}

void
Edge::setData(EdgeType value) {
	edgeData = value;
}


///////////////////
// For `InEdge`. //
///////////////////

InEdge::InEdge(IdType sId, EdgeLocationType eLocation, EdgeType eData)
    : Edge(sId, eLocation, eData) { }

IdType
InEdge::sourceId() {
	return Edge<EdgeType>::otherId;
}

void
InEdge::setSourceId(IdType sId) {
    Edge<EdgeType>::otherId = sId;
}


////////////////////
// For `OutEdge`. //
////////////////////

OutEdge::OutEdge(IdType dId, EdgeLocationType eLocation, EdgeType eData)
    : Edge(dId, eLocation, eData) { }

IdType
OutEdge::destId() {
	return Edge::otherId;
}

void
OutEdge::setDestId(IdType dId) {
    Edge::otherId = dId;
}


#endif // __EDGE_CPP__
