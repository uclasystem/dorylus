#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <set>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <sys/stat.h>
#include <sys/types.h>

typedef unsigned VertexType;

struct BELHeaderType {
    int sizeOfVertexType;
    VertexType numVertices;
    unsigned long long numEdges;
};


int main(int argc, char *argv[]) {
    if (argc != 3) {
        std::cout << "Usage: " << argv[0] << " <GraphBsnapFile> <cscFile>" << std::endl;
        return -1;
    }

    std::string graphPath = argv[1];
    std::string cscPath = argv[2];

    std::ifstream infile(graphPath.c_str(), std::ios::binary);
    if (!infile.good())
        printf("Cannot open graph bsnap file: %s [Reason: %s]\n", graphPath.c_str(), std::strerror(errno));
    assert(infile.good());

    std::cout << "Reading graph bsnap file..." << std::endl;

    BELHeaderType belHeader;
    infile.read((char *) &belHeader, sizeof(belHeader));

    assert(belHeader.sizeOfVertexType == sizeof(unsigned));

    unsigned numVertices = belHeader.numVertices;
    std::cout << "Number of vertices: " << numVertices << std::endl;

    std::set<unsigned> *edgeLists = new std::set<unsigned>[numVertices];
    std::cout << "Number of edges: " << belHeader.numEdges << std::endl;

    unsigned from, to;
    unsigned count;

    while (infile.read((char *) &from, belHeader.sizeOfVertexType)) {
        infile.read((char *) &to, belHeader.sizeOfVertexType);
        edgeLists[to].insert(from);

        ++count;
    }
    infile.close();

    std::cout << "Creating CSC representation..." << std::endl;

    unsigned long long numEdges = 0;
    for (unsigned i = 0; i < numVertices; ++i)
        numEdges += edgeLists[i].size();
    assert(belHeader.numEdges == numEdges);

    unsigned long long *columnPtrs = new unsigned long long[numVertices + 1];
    unsigned *rowIdxs = new unsigned[numEdges];
    unsigned long long edgeItr = 0;
    for (unsigned i = 0; i < numVertices; ++i) {
        columnPtrs[i] = edgeItr;
        std::set<unsigned>::iterator it = edgeLists[i].begin();
        while (it != edgeLists[i].end()) {
            rowIdxs[edgeItr++] = *it;
            ++it;
        }
    }
    columnPtrs[numVertices] = edgeItr;

    unsigned *vsize = new unsigned[numVertices];
    unsigned *vwgt = new unsigned[numVertices];
    for (unsigned i = 0; i < numVertices; ++i) {
        vsize[i] = 1;
        vwgt[i] = 1;
    }

    std::cout << "Partitioning the graph..." << std::endl;

    std::ofstream cscFile;
    cscFile.open(cscPath.c_str());
    cscFile.write((char *)(&numVertices), sizeof(unsigned));
    cscFile.write((char *)(columnPtrs + 1), sizeof(unsigned long long) * numVertices);
    cscFile.write((char *)(&numEdges), sizeof(unsigned long long));
    cscFile.write((char *)rowIdxs, sizeof(unsigned) * numEdges);
    cscFile.close();

    delete[] columnPtrs;
    delete[] rowIdxs;
    delete[] vsize;
    delete[] vwgt;
    delete[] edgeLists;

    return 0;
}
