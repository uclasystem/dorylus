#include <iostream>
#include <cassert>
#include <iterator>
#include "vertex.hpp"

template<typename VertexType, typename EdgeType>
Vertex<VertexType, EdgeType>::Vertex() :
    localIdx(0), globalIdx(0), parentIdx(MAX_IDTYPE), graph(NULL) {
    lock.init();
}

template<typename VertexType, typename EdgeType>
Vertex<VertexType, EdgeType>::~Vertex() {
    lock.destroy();
}

template<typename VertexType, typename EdgeType>
IdType Vertex<VertexType, EdgeType>::localId() {
    return localIdx;
}

template<typename VertexType, typename EdgeType>
IdType Vertex<VertexType, EdgeType>::globalId() {
    return globalIdx;
}

template<typename VertexType, typename EdgeType>
VertexType Vertex<VertexType, EdgeType>::data() {
    lock.readLock();
    VertexType vData = vertexData.back();
    lock.unlock();
    return vData;
}

template<typename VertexType, typename EdgeType>
VertexType Vertex<VertexType, EdgeType>::dataAt(unsigned layer) {
    lock.readLock();
    assert(layer < vertexData.size());
    VertexType vData = vertexData[layer];
    lock.unlock();
    return vData;
}

template<typename VertexType, typename EdgeType>
std::vector<VertexType>& Vertex<VertexType, EdgeType>::dataAll() {
    lock.readLock();
    std::vector<VertexType>& vDataAll = vertexData;
    lock.unlock();
    return vDataAll;
}

template<typename VertexType, typename EdgeType>
void Vertex<VertexType, EdgeType>::setData(VertexType value) {
    lock.writeLock();
    vertexData.back() = value;
    lock.unlock();
}

template<typename VertexType, typename EdgeType>
void Vertex<VertexType, EdgeType>::addData(VertexType value) {
    lock.writeLock();
    vertexData.push_back(value);
    lock.unlock();
}

template<typename VertexType, typename EdgeType>
unsigned Vertex<VertexType, EdgeType>::numInEdges() {
    return inEdges.size();
}

template<typename VertexType, typename EdgeType>
unsigned Vertex<VertexType, EdgeType>::numOutEdges() {
    return outEdges.size();
}

template<typename VertexType, typename EdgeType>
InEdge<EdgeType>& Vertex<VertexType, EdgeType>::getInEdge(unsigned i) {
    return inEdges[i];
}

template<typename VertexType, typename EdgeType>
OutEdge<EdgeType>& Vertex<VertexType, EdgeType>::getOutEdge(unsigned i) {
    return outEdges[i];
}

template<typename VertexType, typename EdgeType>
EdgeType Vertex<VertexType, EdgeType>::getInEdgeData(unsigned i) {
	assert(i < inEdges.size());
	return inEdges[i].data();
}

template<typename VertexType, typename EdgeType>
EdgeType Vertex<VertexType, EdgeType>::getOutEdgeData(unsigned i) {
    assert(false);
    return NULL;
}

template<typename VertexType, typename EdgeType>
VertexType Vertex<VertexType, EdgeType>::getSourceVertexData(unsigned i) {
    assert(i < inEdges.size()); 
    if(inEdges[i].getEdgeLocation() == LOCAL_EDGE_TYPE) {
        return graph->vertices[inEdges[i].sourceId()].data();
    } else {
        return graph->ghostVertices[inEdges[i].sourceId()].data();
    }
}

template<typename VertexType, typename EdgeType>
VertexType Vertex<VertexType, EdgeType>::getSourceVertexDataAt(unsigned i, unsigned layer) {
    assert(i < inEdges.size()); 
    if(inEdges[i].getEdgeLocation() == LOCAL_EDGE_TYPE) {
        return graph->vertices[inEdges[i].sourceId()].dataAt(layer);
    } else {
        return graph->ghostVertices[inEdges[i].sourceId()].dataAt(layer);
    }
}

template<typename VertexType, typename EdgeType>
VertexType Vertex<VertexType, EdgeType>::getDestVertexData(unsigned i) {
    assert(false); 
    return NULL;
}

template<typename VertexType, typename EdgeType>
unsigned Vertex<VertexType, EdgeType>::getSourceVertexGlobalId(unsigned i) {
    assert(i < inEdges.size());
    if(inEdges[i].getEdgeLocation() == LOCAL_EDGE_TYPE) {
        assert(graph->localToGlobalId.find(inEdges[i].sourceId()) != graph->localToGlobalId.end());
        return graph->localToGlobalId[inEdges[i].sourceId()];
    } else {
        return inEdges[i].sourceId();
    }
}

template<typename VertexType, typename EdgeType>
unsigned Vertex<VertexType, EdgeType>::getDestVertexGlobalId(unsigned i) {
    assert(i < outEdges.size());
    if(outEdges[i].getEdgeLocation() == LOCAL_EDGE_TYPE) {
        assert(graph->localToGlobalId.find(outEdges[i].destId()) != graph->localToGlobalId.end());
        return graph->localToGlobalId[outEdges[i].destId()];
    } else {
        return outEdges[i].destId();
    }
}

template<typename VertexType, typename EdgeType>
void Vertex<VertexType, EdgeType>::readLock() {
    lock.readLock();
}

template<typename VertexType, typename EdgeType>
void Vertex<VertexType, EdgeType>::writeLock() {
    lock.writeLock();
}

template<typename VertexType, typename EdgeType>
void Vertex<VertexType, EdgeType>::unlock() {
    lock.unlock();
}

template<typename VertexType, typename EdgeType>
IdType Vertex<VertexType, EdgeType>::parent() {
    return parentIdx;
}

template<typename VertexType, typename EdgeType>
void Vertex<VertexType, EdgeType>::setParent(IdType p) {
    parentIdx = p;
}
