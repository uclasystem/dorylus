#ifndef __VERTEX_CPP__
#define __VERTEX_CPP__


#include <iostream>
#include <cassert>
#include <iterator>
#include "vertex.hpp"


/////////////////////////
// For `Vertex` class. //
/////////////////////////

template<typename VertexType, typename EdgeType>
Vertex<VertexType, EdgeType>::Vertex()
    : localId(0), globalId(0), parentId(MAX_IDTYPE), graph_ptr(NULL) {
    lock.init();
}

template<typename VertexType, typename EdgeType>
Vertex<VertexType, EdgeType>::~Vertex() {
    lock.destroy();
}

template<typename VertexType, typename EdgeType>
IdType
Vertex<VertexType, EdgeType>::getLocalId() {
    return localId;
}

template<typename VertexType, typename EdgeType>
void
Vertex<VertexType, EdgeType>::setLocalId(IdType lvid) {
    localId = lvid;
}

template<typename VertexType, typename EdgeType>
IdType
Vertex<VertexType, EdgeType>::getGlobalId() {
    return globalId;
}

template<typename VertexType, typename EdgeType>
void
Vertex<VertexType, EdgeType>::setGlobalId(IdType gvid) {
    globalId = gvid;
}

template<typename VertexType, typename EdgeType>
VertexType
Vertex<VertexType, EdgeType>::data() {
    lock.readLock();
    VertexType vData = vertexData.back();
    lock.unlock();
    return vData;
}

template<typename VertexType, typename EdgeType>
VertexType
Vertex<VertexType, EdgeType>::dataAt(unsigned layer) {
    lock.readLock();
    assert(layer < vertexData.size());
    VertexType vData = vertexData[layer];
    lock.unlock();
    return vData;
}

template<typename VertexType, typename EdgeType>
std::vector<VertexType>&
Vertex<VertexType, EdgeType>::dataAll() {
    lock.readLock();
    std::vector<VertexType>& vDataAll = vertexData;
    lock.unlock();
    return vDataAll;
}

template<typename VertexType, typename EdgeType>
void
Vertex<VertexType, EdgeType>::setData(VertexType value) {
    lock.writeLock();
    vertexData.back() = value;
    lock.unlock();
}

template<typename VertexType, typename EdgeType>
void
Vertex<VertexType, EdgeType>::addData(VertexType value) {
    lock.writeLock();
    vertexData.push_back(value);
    lock.unlock();
}

template<typename VertexType, typename EdgeType>
VertexLocationType
Vertex<VertexType, EdgeType>::getVertexLocation() {
    return vertexLocation;
}

template<typename VertexType, typename EdgeType>
void
Vertex<VertexType, EdgeType>::setVertexLocation(VertexLocationType loc) {
    vertexLocation = loc;
}

template<typename VertexType, typename EdgeType>
unsigned
Vertex<VertexType, EdgeType>::getNumInEdges() {
    return inEdges.size();
}

template<typename VertexType, typename EdgeType>
unsigned
Vertex<VertexType, EdgeType>::getNumOutEdges() {
    return outEdges.size();
}

template<typename VertexType, typename EdgeType>
InEdge<EdgeType>&
Vertex<VertexType, EdgeType>::getInEdge(unsigned i) {
    return inEdges[i];
}

template<typename VertexType, typename EdgeType>
void
Vertex<VertexType, EdgeType>::addInEdge(InEdge<EdgeType> edge) {
    inEdges.push_back(edge);
}

template<typename VertexType, typename EdgeType>
OutEdge<EdgeType>&
Vertex<VertexType, EdgeType>::getOutEdge(unsigned i) {
    return outEdges[i];
}

template<typename VertexType, typename EdgeType>
void
Vertex<VertexType, EdgeType>::addOutEdge(OutEdge<EdgeType> edge) {
    outEdges.push_back(edge);
}

template<typename VertexType, typename EdgeType>
VertexType
Vertex<VertexType, EdgeType>::getSourceVertexData(unsigned i) {
    assert(i < inEdges.size()); 
    if (inEdges[i].getEdgeLocation() == LOCAL_EDGE_TYPE)
        return graph_ptr->vertices[inEdges[i].sourceId()].data();
    else
        return graph_ptr->ghostVertices[inEdges[i].sourceId()].data();
}

template<typename VertexType, typename EdgeType>
VertexType
Vertex<VertexType, EdgeType>::getSourceVertexDataAt(unsigned i, unsigned layer) {
    assert(i < inEdges.size()); 
    if (inEdges[i].getEdgeLocation() == LOCAL_EDGE_TYPE)
        return graph_ptr->getVertex(inEdges[i].sourceId()).dataAt(layer);
    else
        return graph_ptr->getGhostVertex(inEdges[i].sourceId()).dataAt(layer);
}

template<typename VertexType, typename EdgeType>
unsigned
Vertex<VertexType, EdgeType>::getSourceVertexGlobalId(unsigned i) {
    assert(i < inEdges.size());
    if (inEdges[i].getEdgeLocation() == LOCAL_EDGE_TYPE) {
        assert(graph_ptr->localToGlobalId.find(inEdges[i].sourceId()) != graph_ptr->localToGlobalId.end());
        return graph_ptr->localToGlobalId[inEdges[i].sourceId()];
    } else
        return inEdges[i].sourceId();
}

template<typename VertexType, typename EdgeType>
unsigned
Vertex<VertexType, EdgeType>::getDestVertexGlobalId(unsigned i) {
    assert(i < outEdges.size());
    if (outEdges[i].getEdgeLocation() == LOCAL_EDGE_TYPE) {
        assert(graph_ptr->localToGlobalId.find(outEdges[i].destId()) != graph_ptr->localToGlobalId.end());
        return graph_ptr->localToGlobalId[outEdges[i].destId()];
    } else
        return outEdges[i].destId();
}

template<typename VertexType, typename EdgeType>
Graph<VertexType, EdgeType> *
Vertex<VertexType, EdgeType>::getGraphPtr() {
    return graph_ptr;
}

template<typename VertexType, typename EdgeType>
void
Vertex<VertexType, EdgeType>::setGraphPtr(Graph<VertexType, EdgeType> *ptr) {
    graph_ptr = ptr;
}

template<typename VertexType, typename EdgeType>
IdType
Vertex<VertexType, EdgeType>::getParent() {
    return parentId;
}

template<typename VertexType, typename EdgeType>
void
Vertex<VertexType, EdgeType>::setParent(IdType p) {
    parentId = p;
}

template<typename VertexType, typename EdgeType>
void
Vertex<VertexType, EdgeType>::compactVertex() {
    inEdges.shrink_to_fit();
    outEdges.shrink_to_fit();
}


//////////////////////////////
// For `GhostVertex` class. //
//////////////////////////////

template<typename VertexType>
GhostVertex<VertexType>::GhostVertex()
    : degree(0) {
    lock.init();
}

template<typename VertexType>
GhostVertex<VertexType>::~GhostVertex() {
    lock.destroy();
}

template<typename VertexType>
VertexType
GhostVertex<VertexType>::data() {
    lock.readLock();
    VertexType vData = vertexData.back();
    lock.unlock();
    return vData;
}

template<typename VertexType>
VertexType
GhostVertex<VertexType>::dataAt(unsigned layer) {
    lock.readLock();
    assert(layer < vertexData.size());
    VertexType vData = vertexData[layer];
    lock.unlock();
    return vData;
}

template<typename VertexType>
void
GhostVertex<VertexType>::setData(VertexType value) {
    lock.writeLock();
    vertexData.back() = value;
    lock.unlock();
}

template<typename VertexType>
void
GhostVertex<VertexType>::addData(VertexType value) {
    lock.writeLock();
    vertexData.push_back(value);
    lock.unlock();
}

template<typename VertexType>
void
GhostVertex<VertexType>::addOutEdge(IdType dId) {
    outEdges.push_back(dId);
}

template<typename VertexType>
void
GhostVertex<VertexType>::compactVertex() {
    outEdges.shrink_to_fit();
}

template<typename VertexType>
int32_t
GhostVertex<VertexType>::getDegree() {
    return degree;
}


#endif // __VERTEX_CPP__
