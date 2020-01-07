#include <cerrno>
#include <cmath>
#include <cassert>
#include "dataloader.hpp"
#include "../../common/utils.hpp"


DataLoader::DataLoader(std::string datasetDir, unsigned _nodeId, unsigned _numNodes, bool _undirected) :
                        graphFile(datasetDir + RAWGRAPH_EXT + EDGES_EXT), partsFile(datasetDir + RAWGRAPH_EXT + PARTS_EXT),
                        nodeId(_nodeId), numNodes(_numNodes), undirected(_undirected),
                        forwardDstTables(NULL), backwardDstTables(NULL) {
    char outfileName[50];
    sprintf(outfileName, "graph.%u.bin", nodeId);
    processedGraphFile = datasetDir + std::string(outfileName);

    rawGraph.forwardGhostsList = new std::vector<unsigned>[numNodes];
    rawGraph.backwardGhostsList = new std::vector<unsigned> [numNodes];
}

DataLoader::~DataLoader() {
    if (forwardDstTables) {
        for (unsigned i = 0; i < numNodes; ++i) {
            if (i == nodeId) {
                continue;
            }
            if (forwardDstTables[i]) {
                delete[] forwardDstTables[i];
            }
        }
        delete[] forwardDstTables;
    }
    if (backwardDstTables) {
        for (unsigned i = 0; i < numNodes; ++i) {
            if (i == nodeId) {
                continue;
            }
            if (backwardDstTables[i]) {
                delete[] backwardDstTables[i];
            }
        }
        delete[] backwardDstTables;
    }

    delete[] rawGraph.forwardGhostsList;
    delete[] rawGraph.backwardGhostsList;
}

/**
 *
 * Read in the partition file.
 *
 */
void DataLoader::readPartsFile() {
    std::ifstream infile(partsFile.c_str());
    if (!infile.good())
        printLog(nodeId, "Cannot open patition file: %s [Reason: %s]",
                         partsFile.c_str(), std::strerror(errno));

    assert(infile.good());

    short partId;
    unsigned lvid = 0;
    unsigned gvid = 0;

    std::string line;
    while (std::getline(infile, line)) {
        if (line.size() == 0 || (line[0] < '0' || line[0] > '9'))
            continue;

        std::istringstream iss(line);
        if (!(iss >> partId))
            break;

        rawGraph.appendVertexPartitionId(partId);

        if (partId == nodeId) {
            rawGraph.localToGlobalId[lvid] = gvid;
            rawGraph.globalToLocalId[gvid] = lvid;

            ++lvid;
        }
        ++gvid;
    }

    rawGraph.setNumGlobalVertices(gvid);
    rawGraph.setNumLocalVertices(lvid);
}

/**
 *
 * Process an edge read from the binary snap file.
 *
 */
void DataLoader::processEdge(unsigned &from, unsigned &to) {
    if (rawGraph.getVertexPartitionId(from) == nodeId) {
        unsigned lFromId = rawGraph.globalToLocalId[from];
        unsigned toId;
        EdgeLocationType eLocation;

        unsigned toPartition = rawGraph.getVertexPartitionId(to);
        if (toPartition == nodeId) {
            toId = rawGraph.globalToLocalId[to];
            eLocation = LOCAL_EDGE_TYPE;
        } else {
            toId = to;
            eLocation = REMOTE_EDGE_TYPE;
            rawGraph.getVertex(lFromId).setVertexLocation(BOUNDARY_VERTEX);

            if (!rawGraph.containsOutEdgeGhostVertex(to)) {
                rawGraph.getOutEdgeGhostVertices()[to] = GhostVertex();
            }
            rawGraph.getOutEdgeGhostVertex(to).addAssocEdge(lFromId);

            forwardDstTables[toPartition][lFromId] = true;
        }

        rawGraph.getVertex(lFromId)
                .addOutEdge(OutEdge(toId, eLocation, EdgeType()));
        rawGraph.incrementNumLocalOutEdges();
    }

    if (rawGraph.getVertexPartitionId(to) == nodeId) {
        unsigned lToId = rawGraph.globalToLocalId[to];
        unsigned fromId;
        EdgeLocationType eLocation;

        unsigned fromPartition = rawGraph.getVertexPartitionId(from);
        if (fromPartition == nodeId) {
            fromId = rawGraph.globalToLocalId[from];
            eLocation = LOCAL_EDGE_TYPE;
        } else {
            fromId = from;
            eLocation = REMOTE_EDGE_TYPE;

            if (!rawGraph.containsInEdgeGhostVertex(from)) {
                rawGraph.getInEdgeGhostVertices()[from] = GhostVertex();
            }
            rawGraph.getInEdgeGhostVertex(from).addAssocEdge(lToId);

            backwardDstTables[fromPartition][lToId] = true;
        }

        rawGraph.getVertex(lToId).addInEdge(InEdge(fromId, eLocation, EdgeType()));
        rawGraph.incrementNumLocalInEdges();
    }
}

/**
 *
 * Set the normalization factors on all edges.
 *
 */
void DataLoader::setEdgeNormalizations() {
    for (Vertex &vertex : rawGraph.getVertices()) {
        unsigned vtxDeg = vertex.getNumInEdges() + 1;
        float vtxNorm = std::pow(vtxDeg, -.5);
        vertex.setNormFactor(vtxNorm * vtxNorm);
        for (unsigned i = 0; i < vertex.getNumInEdges(); ++i) {
            InEdge &e = vertex.getInEdge(i);
            unsigned vid = e.getSourceId();
            if (e.getEdgeLocation() == LOCAL_EDGE_TYPE) {
                unsigned srcDeg = rawGraph.getVertex(vid).getNumInEdges() + 1;
                float srcNorm = std::pow(srcDeg, -.5);
                e.setData(srcNorm * vtxNorm);
            } else {
                unsigned ghostDeg = rawGraph.getInEdgeGhostVertex(vid).getDegree() + 1;
                float ghostNorm = std::pow(ghostDeg, -.5);
                e.setData(ghostNorm * vtxNorm);
            }
        }
        for (unsigned i = 0; i < vertex.getNumOutEdges(); ++i) {
            OutEdge &e = vertex.getOutEdge(i);
            unsigned vid = e.getDestId();
            if (e.getEdgeLocation() == LOCAL_EDGE_TYPE) {
                unsigned dstDeg = rawGraph.getVertex(vid).getNumInEdges() + 1;
                float dstNorm = std::pow(dstDeg, -.5);
                e.setData(vtxNorm * dstNorm);
            } else {
                unsigned ghostDeg = rawGraph.getOutEdgeGhostVertex(vid).getDegree() + 1;
                float ghostNorm = std::pow(ghostDeg, -.5);
                e.setData(vtxNorm * ghostNorm);
            }
        }
    }
}

/**
 *
 * Finds the in degree of all ghost vertices.
 *
 */
void DataLoader::findGhostDegrees() {
    std::ifstream infile(graphFile.c_str(), std::ios::binary);
    if (!infile.good())
        printLog(nodeId, "Cannot open BinarySnap file: %s", graphFile.c_str());

    assert(infile.good());

    BSHeaderType bsHeader;
    infile.read((char *)&bsHeader, sizeof(bsHeader));

    unsigned srcdst[2];
    while (infile.read((char *)srcdst, bsHeader.sizeOfVertexType * 2)) {
        if (srcdst[0] == srcdst[1]) {
            continue;
        }

        // YIFAN: we count in degree for both outEdgeGhosts and inEdgeGhosts
        if (rawGraph.containsOutEdgeGhostVertex(srcdst[1])) {
            rawGraph.getOutEdgeGhostVertex(srcdst[1]).incrementDegree();
        }
        if (rawGraph.containsInEdgeGhostVertex(srcdst[1])) {
            rawGraph.getInEdgeGhostVertex(srcdst[1]).incrementDegree();
        }
    }

    infile.close();
}

/**
 *
 * Read and parse the graph from the graph binary snap file.
 *
 */
void DataLoader::preprocess() {
    printLog(nodeId, "Preprocessing... Output to %s", processedGraphFile.c_str());

    // Read in the partition file.
    readPartsFile();

    // Initialize the graph based on the partition info.
    rawGraph.getVertices().resize(rawGraph.getNumLocalVertices());
    for (unsigned i = 0; i < rawGraph.getNumLocalVertices(); ++i) {
        rawGraph.getVertex(i).setLocalId(i);
        rawGraph.getVertex(i).setGlobalId(rawGraph.localToGlobalId[i]);
        rawGraph.getVertex(i).setVertexLocation(INTERNAL_VERTEX);
        rawGraph.getVertex(i).setGraphPtr(&rawGraph);
    }

    // Read in the binary snap edge file.
    std::ifstream infile(graphFile.c_str(), std::ios::binary);
    if (!infile.good())
        printLog(nodeId, "Cannot open BinarySnap file: %s", graphFile.c_str());

    assert(infile.good());

    BSHeaderType bSHeader;
    infile.read((char *)&bSHeader, sizeof(bSHeader));
    assert(bSHeader.sizeOfVertexType == sizeof(unsigned));

    forwardDstTables = new bool *[numNodes];
    backwardDstTables = new bool *[numNodes];
    for (unsigned i = 0; i < numNodes; ++i) {
        if (i == nodeId) {
            continue;
        }
        forwardDstTables[i] = new bool[rawGraph.getNumLocalVertices()];
        memset(forwardDstTables[i], 0,
                     sizeof(bool) * rawGraph.getNumLocalVertices());
        backwardDstTables[i] = new bool[rawGraph.getNumLocalVertices()];
        memset(backwardDstTables[i], 0,
                     sizeof(bool) * rawGraph.getNumLocalVertices());
    }

    // Loop through all edges and process them.
    unsigned srcdst[2];
    while (infile.read((char *)srcdst, bSHeader.sizeOfVertexType * 2)) {
        if (srcdst[0] == srcdst[1])
            continue;

        processEdge(srcdst[0], srcdst[1]);
        if (undirected)
            processEdge(srcdst[1], srcdst[0]);
        rawGraph.incrementNumGlobalEdges();
    }

    for (unsigned i = 0; i < numNodes; ++i) {
        if (i == nodeId) {
            continue;
        }
        rawGraph.forwardGhostsList[i].reserve(rawGraph.getNumLocalVertices());
        rawGraph.backwardGhostsList[i].reserve(rawGraph.getNumLocalVertices());
        for (unsigned j = 0; j < rawGraph.getNumLocalVertices(); ++j) {
            if (forwardDstTables[i][j]) {
                rawGraph.forwardGhostsList[i].push_back(j);
            }
            if (backwardDstTables[i][j]) {
                rawGraph.backwardGhostsList[i].push_back(j);
            }
        }
        rawGraph.forwardGhostsList[i].shrink_to_fit();
        rawGraph.backwardGhostsList[i].shrink_to_fit();
        delete[] forwardDstTables[i];
        delete[] backwardDstTables[i];
        forwardDstTables[i] = NULL;
        backwardDstTables[i] = NULL;
    }
    delete[] forwardDstTables;
    delete[] backwardDstTables;
    forwardDstTables = NULL;
    backwardDstTables = NULL;

    infile.close();

    // Extra works added.
    rawGraph.setNumInEdgeGhostVertices(rawGraph.getInEdgeGhostVertices().size());
    rawGraph.setNumOutEdgeGhostVertices(rawGraph.getOutEdgeGhostVertices().size());
    findGhostDegrees();
    setEdgeNormalizations();

    // Set a local index for all ghost vertices along the way. This index is used
    // for indexing within the ghost data arrays.
    unsigned ghostCount = rawGraph.getNumLocalVertices();
    for (auto it = rawGraph.getInEdgeGhostVertices().begin();
             it != rawGraph.getInEdgeGhostVertices().end(); it++) {
        it->second.setLocalId(ghostCount++);
    }
    ghostCount = rawGraph.getNumLocalVertices();
    for (auto it = rawGraph.getOutEdgeGhostVertices().begin();
             it != rawGraph.getOutEdgeGhostVertices().end(); it++) {
        it->second.setLocalId(ghostCount++);
    }

    rawGraph.forwardAdj.init(rawGraph);
    rawGraph.backwardAdj.init(rawGraph);

    rawGraph.dump(processedGraphFile, numNodes);

    printLog(nodeId, "Finish preprocessing!");
}
