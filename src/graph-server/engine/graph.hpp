#ifndef __GRAPH_HPP__
#define __GRAPH_HPP__


#include <vector>
#include <map>
#include "../parallel/lock.hpp"
#include "../utils/utils.hpp"
#include "vertex.hpp"
#include "edge.hpp"

class Graph;
class RawGraph;

template<typename T>
class CSCMatrix {
public:
    CSCMatrix() : columnCnt(0), nnz(0), values(NULL), locations(NULL), columnPtrs(NULL), rowIdxs(NULL) {};
    ~CSCMatrix() {
        if (values)     { delete[] values; }
        if (locations)  { delete[] locations; }
        if (columnPtrs) { delete[] columnPtrs; }
        if (rowIdxs)    { delete[] rowIdxs; }
    };
    void init(RawGraph &rgraph);

    unsigned columnCnt;
    unsigned long long nnz;         // number of non-zero elements
    T *values;                      // non-zero elements
    char *locations;                // edge locations vector
    unsigned long long *columnPtrs; // pointers to the start of each column
    unsigned *rowIdxs;              // indices of nz elements in each column
};

template<typename T>
class CSRMatrix {
public:
    CSRMatrix() : rowCnt(0), nnz(0), values(NULL), locations(NULL), rowPtrs(NULL), columnIdxs(NULL) {};
    ~CSRMatrix() {
        if (values)     { delete[] values; }
        if (locations)  { delete[] locations; }
        if (rowPtrs)    { delete[] rowPtrs; }
        if (columnIdxs) { delete[] columnIdxs; }
    };
    void init(RawGraph &rgraph);

    unsigned rowCnt;
    unsigned long long nnz;      // number of non-zero elements
    T *values;                   // non-zero elements
    char *locations;             // edge locations vector
    unsigned long long *rowPtrs; // pointers to the start of each row
    unsigned *columnIdxs;        // indices of nz elements in each row
};

/**
 *
 * Class of a graph, composed of vertices and directed edges.
 *
 */
class Graph {
public:
    void init(std::string graphFile);
    bool containsVtx(unsigned gvid);
    bool containsSrcGhostVtx(unsigned gvid);
    bool containsDstGhostVtx(unsigned gvid);

    void print();
    // members
    // vtx cnt
    unsigned localVtxCnt;
    unsigned globalVtxCnt;
    unsigned srcGhostCnt;
    unsigned dstGhostCnt;
    // edg cnt
    unsigned long long localInEdgeCnt = 0;
    unsigned long long localOutEdgeCnt = 0;
    unsigned long long globalEdgeCnt = 0;
    // local vertices
    std::vector<unsigned> localToGlobalId;
    std::map<unsigned, unsigned> globaltoLocalId;
    std::vector<EdgeType> vtxDataVec;
    // local vertex outgoing destinations
    std::vector<std::vector<unsigned>> forwardLocalVtxDsts;
    std::vector<std::vector<unsigned>> backwardLocalVtxDsts;

    // Outoing dests for pipelining
    std::map<unsigned, std::vector<unsigned>> forwardGhostMap;
    std::map<unsigned, std::vector<unsigned>> backwardGhostMap;

    // incoming edge ghost vertices
    std::map<unsigned, unsigned> srcGhostVtcs;
    // std::vector<unsigned> srcGhostDataVec;
    // outgoing edge ghost vertices
    std::map<unsigned, unsigned> dstGhostVtcs;
    // std::vector<unsigned> dstGhostDataVec;
    // ajacency matrices
    CSCMatrix<EdgeType> forwardAdj;
    CSRMatrix<EdgeType> backwardAdj;
};

class RawGraph {
public:

    std::vector<Vertex>& getVertices() { return vertices; }
    Vertex& getVertex(unsigned lvid);
    Vertex& getVertexByGlobal(unsigned gvid);
    bool containsVertex(unsigned gvid);   // Contain searches with global ID.

    std::map<unsigned, GhostVertex>& getInEdgeGhostVertices()  { return inEdgeGhostVertices; }
    std::map<unsigned, GhostVertex>& getOutEdgeGhostVertices() { return outEdgeGhostVertices; }
    GhostVertex& getInEdgeGhostVertex(unsigned gvid);
    GhostVertex& getOutEdgeGhostVertex(unsigned gvid);
    bool containsInEdgeGhostVertex(unsigned gvid);
    bool containsOutEdgeGhostVertex(unsigned gvid);

    unsigned getNumLocalVertices() { return numLocalVertices; }
    void setNumLocalVertices(unsigned num) { numLocalVertices = num; }
    unsigned getNumGlobalVertices() { return numGlobalVertices; }
    void setNumGlobalVertices(unsigned num) { numGlobalVertices = num; }
    unsigned getNumInEdgeGhostVertices()  { return numInEdgeGhostVertices; }
    unsigned getNumOutEdgeGhostVertices() { return numOutEdgeGhostVertices; }
    void setNumInEdgeGhostVertices(unsigned num)  { numInEdgeGhostVertices = num; }
    void setNumOutEdgeGhostVertices(unsigned num) { numOutEdgeGhostVertices = num; }

    unsigned long long getNumLocalInEdges() { return numLocalInEdges; }
    void incrementNumLocalInEdges() { ++numLocalInEdges; }
    unsigned long long getNumLocalOutEdges() { return numLocalOutEdges; }
    void incrementNumLocalOutEdges() { ++numLocalOutEdges; }
    unsigned long long getNumGlobalEdges() { return numGlobalEdges; }
    void incrementNumGlobalEdges() { ++numGlobalEdges; }

    short getVertexPartitionId(unsigned vid) { return vertexPartitionIds[vid]; }
    void appendVertexPartitionId(short pid) { vertexPartitionIds.push_back(pid); }

    void compactGraph();
    void dump(std::string filename, unsigned numNodes);

    std::map<unsigned, unsigned> globalToLocalId;
    std::map<unsigned, unsigned> localToGlobalId;

    std::vector<unsigned> *forwardGhostsList;
    std::vector<unsigned> *backwardGhostsList;

    CSCMatrix<EdgeType> forwardAdj;
    CSRMatrix<EdgeType> backwardAdj;

private:

    std::vector<Vertex> vertices;
    std::map<unsigned, GhostVertex> inEdgeGhostVertices;
    std::map<unsigned, GhostVertex> outEdgeGhostVertices;

    unsigned numLocalVertices;
    unsigned numGlobalVertices;
    unsigned numInEdgeGhostVertices;
    unsigned numOutEdgeGhostVertices;

    // Local Edge: the src vertex locates on the local machine.
    unsigned long long numLocalInEdges = 0;
    unsigned long long numLocalOutEdges = 0;
    unsigned long long numGlobalEdges = 0;

    std::vector<short> vertexPartitionIds;
};


template<typename T>
void CSCMatrix<T>::init(RawGraph &rgraph) {
    columnCnt = rgraph.getNumLocalVertices();
    nnz = rgraph.getNumLocalInEdges();

    values = new T[nnz];
    locations = new char[nnz];
    columnPtrs = new unsigned long long[columnCnt + 1];
    rowIdxs = new unsigned [nnz];

    columnPtrs[0] = 0;
    unsigned long long edgItr = 0;
    for (unsigned lvid = 0; lvid < columnCnt; ++lvid) {
        Vertex &v = rgraph.getVertex(lvid);
        const unsigned edgsCnt = v.getNumInEdges();
        columnPtrs[lvid + 1] = columnPtrs[lvid] + edgsCnt;
        for (unsigned veItr = 0; veItr < edgsCnt; ++veItr) {
            InEdge &vie = v.getInEdge(veItr);
            values[edgItr] = static_cast<T>(vie.getData());
            rowIdxs[edgItr] = v.getSourceVertexLocalId(veItr);
            ++edgItr;
        }
    }
}

template<typename T>
void CSRMatrix<T>::init(RawGraph &rgraph) {
    rowCnt = rgraph.getNumLocalVertices();
    nnz = rgraph.getNumLocalOutEdges();

    values = new T[nnz];
    locations = new char[nnz];
    rowPtrs = new unsigned long long[rowCnt + 1];
    columnIdxs = new unsigned[nnz];

    rowPtrs[0] = 0;
    unsigned long long edgItr = 0;
    for (unsigned lvid = 0; lvid < rowCnt; ++lvid) {
        Vertex &v = rgraph.getVertex(lvid);
        const unsigned edgsCnt = v.getNumOutEdges();
        rowPtrs[lvid + 1] = rowPtrs[lvid] + edgsCnt;
        for (unsigned veItr = 0; veItr < edgsCnt; ++veItr) {
            OutEdge &voe = v.getOutEdge(veItr);
            values[edgItr] = static_cast<T>(voe.getData());
            columnIdxs[edgItr] = v.getDestVertexLocalId(veItr);
            ++edgItr;
        }
    }
}

#endif //__GRAPH_HPP__
