#include <cassert>
#include "graph.hpp"
#include <fstream>
#include <iostream>
#include <sys/stat.h>

void Graph::init(std::string graphFile) {
    std::ifstream infile(graphFile.c_str(), std::ios::binary);
    if (!infile.good()) {
        std::cout << "Cannot open input file: " << graphFile << ", [Reason: " << std::strerror(errno) << "]" << std::endl;
        return;
    }
    assert(infile.good());
    // vertex count
    infile.read(reinterpret_cast<char *>(&localVtxCnt), sizeof(unsigned));
    infile.read(reinterpret_cast<char *>(&globalVtxCnt), sizeof(unsigned));
    infile.read(reinterpret_cast<char *>(&srcGhostCnt), sizeof(unsigned));
    infile.read(reinterpret_cast<char *>(&dstGhostCnt), sizeof(unsigned));
    // edge count
    infile.read(reinterpret_cast<char *>(&localInEdgeCnt), sizeof(unsigned long long));
    infile.read(reinterpret_cast<char *>(&localOutEdgeCnt), sizeof(unsigned long long));
    infile.read(reinterpret_cast<char *>(&globalEdgeCnt), sizeof(unsigned long long));

    localToGlobalId.resize(localVtxCnt);
    // local vertex global IDs
    infile.read(reinterpret_cast<char *>(localToGlobalId.data()), sizeof(unsigned) * localVtxCnt);
    // construct mapping of local vertex global IDs to local IDs
    for (unsigned lvid = 0; lvid < localVtxCnt; ++lvid) {
        globaltoLocalId[localToGlobalId[lvid]] = lvid;
    }
    vtxDataVec.resize(localVtxCnt);
    // vertex data (normFactor for GCN)
    infile.read(reinterpret_cast<char *>(vtxDataVec.data()), sizeof(EdgeType) * localVtxCnt);
    // mapping of src ghost vertex global IDs to local IDs
    for (unsigned i = 0; i < srcGhostCnt; ++i) {
        unsigned gvid = 0;
        unsigned lvid = 0;
        infile.read(reinterpret_cast<char *>(&gvid), sizeof(unsigned));
        infile.read(reinterpret_cast<char *>(&lvid), sizeof(unsigned));
        srcGhostVtcs[gvid] = lvid;
    }
    // mapping of dst ghost vertex global IDs to local IDs
    for (unsigned i = 0; i < dstGhostCnt; ++i) {
        unsigned gvid = 0;
        unsigned lvid = 0;
        infile.read(reinterpret_cast<char *>(&gvid), sizeof(unsigned));
        infile.read(reinterpret_cast<char *>(&lvid), sizeof(unsigned));
        dstGhostVtcs[gvid] = lvid;
    }
    // destination of local vertices during forward
    unsigned numNodes = 0;
    infile.read(reinterpret_cast<char *>(&numNodes), sizeof(unsigned));
    for (unsigned i = 0; i < numNodes; ++i) {
        unsigned size = 0;
        infile.read(reinterpret_cast<char *>(&size), sizeof(unsigned));
        forwardLocalVtxDsts.push_back(std::vector<unsigned>(size));
        infile.read(reinterpret_cast<char *>(forwardLocalVtxDsts[i].data()), sizeof(unsigned) * size);
    }
    // destination of local vertices during backward
    for (unsigned i = 0; i < numNodes; ++i) {
        unsigned size = 0;
        infile.read(reinterpret_cast<char *>(&size), sizeof(unsigned));
        backwardLocalVtxDsts.push_back(std::vector<unsigned>(size));
        infile.read(reinterpret_cast<char *>(backwardLocalVtxDsts[i].data()), sizeof(unsigned) * size);
    }

    // Create alternate representation of local vertex destinations for pipelined version
    for (unsigned nid = 0; nid < numNodes; ++nid) {
        std::vector<unsigned>& vids = forwardLocalVtxDsts[nid];
        for (unsigned i = 0; i < vids.size(); ++i) {
            if (forwardGhostMap.find(vids[i]) == forwardGhostMap.end()) {
                forwardGhostMap[vids[i]] = std::vector<unsigned>();
            }

            forwardGhostMap[vids[i]].push_back(nid);
        }
    }

    for (unsigned nid = 0; nid < numNodes; ++nid) {
        std::vector<unsigned>& vids = backwardLocalVtxDsts[nid];
        for (unsigned i = 0; i < vids.size(); ++i) {
            if (backwardGhostMap.find(vids[i]) == backwardGhostMap.end()) {
                backwardGhostMap[vids[i]] = std::vector<unsigned>();
            }

            backwardGhostMap[vids[i]].push_back(nid);
        }
    }

    // CSC representation of graph
    infile.read(reinterpret_cast<char *>(&forwardAdj.columnCnt), sizeof(unsigned));
    infile.read(reinterpret_cast<char *>(&forwardAdj.nnz), sizeof(unsigned long long));
    forwardAdj.values = new EdgeType[forwardAdj.nnz];
    forwardAdj.locations = new char[forwardAdj.nnz];
    forwardAdj.columnPtrs = new unsigned long long[localVtxCnt + 1];
    forwardAdj.rowIdxs = new unsigned[forwardAdj.nnz];
    infile.read(reinterpret_cast<char *>(forwardAdj.values), sizeof(EdgeType) * forwardAdj.nnz);
    // infile.read(reinterpret_cast<char *>(forwardAdj.locations), sizeof(char) * forwardAdj.nnz);
    infile.read(reinterpret_cast<char *>(forwardAdj.columnPtrs), sizeof(unsigned long long) * (localVtxCnt + 1));
    infile.read(reinterpret_cast<char *>(forwardAdj.rowIdxs), sizeof(unsigned) * forwardAdj.nnz);

    // CSR representation of grpah
    infile.read(reinterpret_cast<char *>(&backwardAdj.rowCnt), sizeof(unsigned));
    infile.read(reinterpret_cast<char *>(&backwardAdj.nnz), sizeof(unsigned long long));
    backwardAdj.values = new EdgeType[backwardAdj.nnz];
    backwardAdj.locations = new char[backwardAdj.nnz];
    backwardAdj.rowPtrs = new unsigned long long[localVtxCnt + 1];
    backwardAdj.columnIdxs = new unsigned[backwardAdj.nnz];
    infile.read(reinterpret_cast<char *>(backwardAdj.values), sizeof(EdgeType) * backwardAdj.nnz);
    // infile.read(reinterpret_cast<char *>(backwardAdj.locations), sizeof(char) * backwardAdj.nnz);
    infile.read(reinterpret_cast<char *>(backwardAdj.rowPtrs), sizeof(unsigned long long) * (localVtxCnt + 1));
    infile.read(reinterpret_cast<char *>(backwardAdj.columnIdxs), sizeof(unsigned) * backwardAdj.nnz);

    infile.close();
}

bool Graph::containsVtx(unsigned gvid) {
    return globaltoLocalId.find(gvid) != globaltoLocalId.end();
}

bool Graph::containsSrcGhostVtx(unsigned gvid) {
    return srcGhostVtcs.find(gvid) != srcGhostVtcs.end();
}

bool Graph::containsDstGhostVtx(unsigned gvid) {
    return dstGhostVtcs.find(gvid) != dstGhostVtcs.end();
}

void Graph::print() {
    fprintf(stderr, "%d %d %d %d; %lld %lld %lld; %lld %lld\n",
            localVtxCnt, globalVtxCnt, srcGhostCnt, dstGhostCnt,
            localInEdgeCnt, localOutEdgeCnt, globalEdgeCnt,
            forwardAdj.nnz, backwardAdj.nnz);
}

Vertex&
RawGraph::getVertex(unsigned lvid) {
    assert(lvid < vertices.size());
    return vertices[lvid];
}

Vertex&
RawGraph::getVertexByGlobal(unsigned gvid) {
    assert(globalToLocalId.find(gvid) != globalToLocalId.end());
    return vertices[globalToLocalId[gvid]];
}

bool
RawGraph::containsVertex(unsigned gvid) {
    return globalToLocalId.find(gvid) != globalToLocalId.end();
}

GhostVertex&
RawGraph::getInEdgeGhostVertex(unsigned gvid) {
    assert(inEdgeGhostVertices.find(gvid) != inEdgeGhostVertices.end());
    return inEdgeGhostVertices[gvid];
}

GhostVertex&
RawGraph::getOutEdgeGhostVertex(unsigned gvid) {
    assert(outEdgeGhostVertices.find(gvid) != outEdgeGhostVertices.end());
    return outEdgeGhostVertices[gvid];
}

bool
RawGraph::containsInEdgeGhostVertex(unsigned gvid) {
    return inEdgeGhostVertices.find(gvid) != inEdgeGhostVertices.end();
}

bool
RawGraph::containsOutEdgeGhostVertex(unsigned gvid) {
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
RawGraph::compactGraph() {
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

void
RawGraph::dump(std::string filename, unsigned numNodes) {
    std::ofstream outfile(filename, std::ofstream::binary);
    if (!outfile.good()) {
        std::cout << "Cannot open output file:" << filename << ", [Reason: " << std::strerror(errno) << "]" << std::endl;
        return;
    }
    // vertex count (local/global/incoming ghost/outgoing ghost)
    outfile.write(reinterpret_cast<const char*>(&numLocalVertices), sizeof(numLocalVertices));
    outfile.write(reinterpret_cast<const char*>(&numGlobalVertices), sizeof(numGlobalVertices));
    outfile.write(reinterpret_cast<const char *>(&numInEdgeGhostVertices), sizeof(numInEdgeGhostVertices));
    outfile.write(reinterpret_cast<const char *>(&numOutEdgeGhostVertices), sizeof(numOutEdgeGhostVertices));
    // edge count (local incoming/local outgoing/global)
    outfile.write(reinterpret_cast<const char*>(&numLocalInEdges), sizeof(numLocalInEdges));
    outfile.write(reinterpret_cast<const char*>(&numLocalOutEdges), sizeof(numLocalOutEdges));
    outfile.write(reinterpret_cast<const char*>(&numGlobalEdges), sizeof(numGlobalEdges));
    // global IDs of local vertices
    for (unsigned i = 0; i < numLocalVertices; ++i) {
        unsigned gvid = localToGlobalId[i];
        outfile.write(reinterpret_cast<const char *>(&gvid), sizeof(unsigned));
    }
    // normFactors of local vertices
    for (unsigned i = 0; i < numLocalVertices; ++i) {
        EdgeType normFactor = vertices[i].getNormFactor();
        outfile.write(reinterpret_cast<const char *>(&normFactor), sizeof(normFactor));
    }
    // mapping of incoming ghost's global ID to local ID
    for (auto &itr: inEdgeGhostVertices) {
        unsigned gvid = itr.first;
        unsigned lvid = itr.second.getLocalId();
        outfile.write(reinterpret_cast<const char *>(&gvid), sizeof(unsigned));
        outfile.write(reinterpret_cast<const char *>(&lvid), sizeof(unsigned));
    }
    // mapping of outgoing ghost's global ID to local ID
    for (auto &itr : outEdgeGhostVertices) {
        unsigned gvid = itr.first;
        unsigned lvid = itr.second.getLocalId();
        outfile.write(reinterpret_cast<const char *>(&gvid), sizeof(unsigned));
        outfile.write(reinterpret_cast<const char *>(&lvid), sizeof(unsigned));
    }
    outfile.write(reinterpret_cast<const char *>(&numNodes), sizeof(unsigned));
    // local vertices send out destinations during forward
    for (unsigned i = 0; i < numNodes; ++i) {
        unsigned size = forwardGhostsList[i].size();
        outfile.write(reinterpret_cast<const char *>(&size), sizeof(unsigned));
        outfile.write(reinterpret_cast<const char *>(forwardGhostsList[i].data()), sizeof(unsigned) * size);
    }
    // local vertices send out destinations during backward
    for (unsigned i = 0; i < numNodes; ++i) {
        unsigned size = backwardGhostsList[i].size();
        outfile.write(reinterpret_cast<const char *>(&size), sizeof(unsigned));
        outfile.write(reinterpret_cast<const char *>(backwardGhostsList[i].data()), sizeof(unsigned) * size);
    }

    // CSC representation of graph
    outfile.write(reinterpret_cast<const char *>(&forwardAdj.columnCnt), sizeof(unsigned));
    outfile.write(reinterpret_cast<const char *>(&forwardAdj.nnz), sizeof(unsigned long long));
    outfile.write(reinterpret_cast<const char *>(forwardAdj.values), sizeof(EdgeType) * forwardAdj.nnz);
    // outfile.write(reinterpret_cast<const char *>(forwardAdj.locations), sizeof(char) * forwardAdj.nnz);
    outfile.write(reinterpret_cast<const char *>(forwardAdj.columnPtrs), sizeof(unsigned long long) * (numLocalVertices + 1));
    outfile.write(reinterpret_cast<const char *>(forwardAdj.rowIdxs), sizeof(unsigned) * forwardAdj.nnz);

    // CSR representation of graph
    outfile.write(reinterpret_cast<const char *>(&backwardAdj.rowCnt), sizeof(unsigned));
    outfile.write(reinterpret_cast<const char *>(&backwardAdj.nnz), sizeof(unsigned long long));
    outfile.write(reinterpret_cast<const char *>(backwardAdj.values), sizeof(EdgeType) * backwardAdj.nnz);
    // outfile.write(reinterpret_cast<const char *>(backwardAdj.locations), sizeof(char) * backwardAdj.nnz);
    outfile.write(reinterpret_cast<const char *>(backwardAdj.rowPtrs), sizeof(unsigned long long) * (numLocalVertices + 1));
    outfile.write(reinterpret_cast<const char *>(backwardAdj.columnIdxs), sizeof(unsigned) * backwardAdj.nnz);

    outfile.close();
    // set file permission to 777 to allow accesses from other users
    chmod(filename.c_str(), S_IRWXU | S_IRWXG | S_IRWXO);
}
