#include <iostream>
#include <cassert>
#include <iterator>
#include "vertex.hpp"
#include "graph.hpp"


/////////////////////////
// For `Vertex` class. //
/////////////////////////

Vertex::Vertex()
    : localId(0), globalId(0), parentId(MAX_IDTYPE), graph_ptr(NULL) {
    lock.init();
}

Vertex::~Vertex() {
    lock.destroy();
}

IdType
Vertex::getLocalId() {
    return localId;
}

void
Vertex::setLocalId(IdType lvid) {
    localId = lvid;
}

IdType
Vertex::getGlobalId() {
    return globalId;
}

void
Vertex::setGlobalId(IdType gvid) {
    globalId = gvid;
}

VertexType
Vertex::data() {
    lock.readLock();
    VertexType vData = vertexData.back();
    lock.unlock();
    return vData;
}

VertexType
Vertex::dataAt(unsigned layer) {
    lock.readLock();
    assert(layer < vertexData.size());
    VertexType vData = vertexData[layer];
    lock.unlock();
    return vData;
}

std::vector<VertexType>&
Vertex::dataAll() {
    lock.readLock();
    std::vector<VertexType>& vDataAll = vertexData;
    lock.unlock();
    return vDataAll;
}

void
Vertex::setData(VertexType value) {
    lock.writeLock();
    if (vertexData.empty())
        vertexData.push_back(value);
    else
        vertexData.back() = value;
    lock.unlock();
}

void
Vertex::addData(VertexType value) {
    lock.writeLock();
    vertexData.push_back(value);
    lock.unlock();
}

VertexLocationType
Vertex::getVertexLocation() {
    return vertexLocation;
}

void
Vertex::setVertexLocation(VertexLocationType loc) {
    vertexLocation = loc;
}

unsigned
Vertex::getNumInEdges() {
    return inEdges.size();
}

unsigned
Vertex::getNumOutEdges() {
    return outEdges.size();
}

InEdge&
Vertex::getInEdge(unsigned i) {
    return inEdges[i];
}

void
Vertex::addInEdge(InEdge edge) {
    inEdges.push_back(edge);
}

OutEdge&
Vertex::getOutEdge(unsigned i) {
    return outEdges[i];
}

void
Vertex::addOutEdge(OutEdge edge) {
    outEdges.push_back(edge);
}

VertexType
Vertex::getSourceVertexData(unsigned i) {
    assert(i < inEdges.size()); 
    if (inEdges[i].getEdgeLocation() == LOCAL_EDGE_TYPE)
        return graph_ptr->getVertex(inEdges[i].sourceId()).data();
    else
        return graph_ptr->getGhostVertices(inEdges[i].sourceId()).data();
}

VertexType
Vertex::getSourceVertexDataAt(unsigned i, unsigned layer) {
    assert(i < inEdges.size()); 
    if (inEdges[i].getEdgeLocation() == LOCAL_EDGE_TYPE)
        return graph_ptr->getVertex(inEdges[i].sourceId()).dataAt(layer);
    else
        return graph_ptr->getGhostVertex(inEdges[i].sourceId()).dataAt(layer);
}

unsigned
Vertex::getSourceVertexGlobalId(unsigned i) {
    assert(i < inEdges.size());
    if (inEdges[i].getEdgeLocation() == LOCAL_EDGE_TYPE) {
        assert(graph_ptr->localToGlobalId.find(inEdges[i].sourceId()) != graph_ptr->localToGlobalId.end());
        return graph_ptr->localToGlobalId[inEdges[i].sourceId()];
    } else
        return inEdges[i].sourceId();
}

unsigned
Vertex::getDestVertexGlobalId(unsigned i) {
    assert(i < outEdges.size());
    if (outEdges[i].getEdgeLocation() == LOCAL_EDGE_TYPE) {
        assert(graph_ptr->localToGlobalId.find(outEdges[i].destId()) != graph_ptr->localToGlobalId.end());
        return graph_ptr->localToGlobalId[outEdges[i].destId()];
    } else
        return outEdges[i].destId();
}

Graph *
Vertex::getGraphPtr() {
    return graph_ptr;
}

void
Vertex::setGraphPtr(Graph *ptr) {
    graph_ptr = ptr;
}

IdType
Vertex::getParent() {
    return parentId;
}

void
Vertex::setParent(IdType p) {
    parentId = p;
}

void
Vertex::compactVertex() {
    inEdges.shrink_to_fit();
    outEdges.shrink_to_fit();
}


//////////////////////////////
// For `GhostVertex` class. //
//////////////////////////////

GhostVertex::GhostVertex()
    : degree(0) {
    lock.init();
}

GhostVertex::~GhostVertex() {
    lock.destroy();
}

VertexType
GhostVertex::data() {
    lock.readLock();
    VertexType vData = vertexData.back();
    lock.unlock();
    return vData;
}

VertexType
GhostVertex::dataAt(unsigned layer) {
    lock.readLock();
    assert(layer < vertexData.size());
    VertexType vData = vertexData[layer];
    lock.unlock();
    return vData;
}

void
GhostVertex::setData(VertexType value) {
    lock.writeLock();
    if (vertexData.empty())
        vertexData.push_back(value);
    else
        vertexData.back() = value;
    lock.unlock();
}

void
GhostVertex::addData(VertexType value) {
    lock.writeLock();
    vertexData.push_back(value);
    lock.unlock();
}

void
GhostVertex::addOutEdge(IdType dId) {
    outEdges.push_back(dId);
}

void
GhostVertex::compactVertex() {
    outEdges.shrink_to_fit();
}

int32_t
GhostVertex::getDegree() {
    return degree;
}
