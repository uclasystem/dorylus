#ifndef __EDGE_CPP__
#define __EDGE_CPP__


#include <cstring>
#include "edge.hpp"


////////////////////////////
// For base class `Edge`. //
////////////////////////////

template<typename EdgeType>
Edge<EdgeType>::Edge(IdType oId, EdgeLocationType eLocation, EdgeType eData)
    : otherId(oId), edgeData(eData), edgeLocation(eLocation) { }

template<typename EdgeType>
EdgeType
Edge<EdgeType>::data() {
    return edgeData;
}

template<typename EdgeType>
EdgeLocationType
Edge<EdgeType>::getEdgeLocation() {
    return edgeLocation;
}

template<typename EdgeType>
void
Edge<EdgeType>::setEdgeLocation(EdgeLocationType eLoc) {
    edgeLocation = eLoc;
}

template<typename EdgeType>
void
Edge<EdgeType>::setData(EdgeType value) {
	edgeData = value;
}


///////////////////
// For `InEdge`. //
///////////////////

template<typename EdgeType>
InEdge<EdgeType>::InEdge(IdType sId, EdgeLocationType eLocation, EdgeType eData)
    : Edge<EdgeType>(sId, eLocation, eData) { }

template<typename EdgeType>
IdType
InEdge<EdgeType>::sourceId() {
	return Edge<EdgeType>::otherId;
}

template<typename EdgeType>
void
InEdge<EdgeType>::setSourceId(IdType sId) {
    Edge<EdgeType>::otherId = sId;
}


////////////////////
// For `OutEdge`. //
////////////////////

template<typename EdgeType>
OutEdge<EdgeType>::OutEdge(IdType dId, EdgeLocationType eLocation, EdgeType eData)
    : Edge<EdgeType>(dId, eLocation, eData) { }

template<typename EdgeType>
IdType
OutEdge<EdgeType>::destId() {
	return Edge<EdgeType>::otherId;
}

template<typename EdgeType>
void
OutEdge<EdgeType>::setDestId(IdType dId) {
    Edge<EdgeType>::otherId = dId;
}


#endif // __EDGE_CPP__
