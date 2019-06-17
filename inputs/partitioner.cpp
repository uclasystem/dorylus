#include <iostream>
#include <fstream>
#include <sstream>
#include <set>
#include <cassert>
#include <cstdlib>
#include <sys/stat.h>
#include <sys/types.h>

#include "metis.h"

#define BASE_PATH "./"
#define PARTS_PATH BASE_PATH "parts_"
#define COMM_EXT ".comm"
#define PART_EXT ".parts"

typedef unsigned VertexType;

struct BELHeaderType {
    int sizeOfVertexType;
//    int sizeOfCountType;
    VertexType numVertices;
    unsigned long long numEdges;
};

int main(int argc, char* argv[])
{
    if(argc < 4) {
        fprintf(stderr, "NEVER INVOKE ME DIRECTLY! USE partitioner.sh SCRIPT WHICH WILL INTERNALLY INVOKE ME!\n");
        fprintf(stderr, "Dude! Invoke like this: <graph-file> <num-vertices> <num-partitions>\n");
        exit(-1);
    }

    std::string graphName = argv[1];
    idx_t nparts = atoi(argv[3]);

    std::string partsDir = std::string(PARTS_PATH) + argv[3] + "/";
    mkdir(partsDir.c_str(), 0777);

    fprintf(stderr, "Reading input graph ...\n");
    std::string graphPath = BASE_PATH + graphName;
    std::ifstream infile(graphPath.c_str(), std::ios::binary);

    if(!infile.good()) {
        fprintf(stderr, "Cannot open %s\n", graphPath.c_str());
        abort();
    }

	std::cerr << "READING HEADER" << std::endl;
    BELHeaderType belHeader;
    infile.read((char*) &belHeader, sizeof(belHeader));

    assert(belHeader.sizeOfVertexType == sizeof(unsigned));
//    assert(belHeader.sizeOfCountType == sizeof(unsigned));
	idx_t nvtxs = belHeader.numVertices;
	std::cout << "NUM VERTS: " << nvtxs << std::endl;
    std::set<idx_t>* edgeLists = new std::set<idx_t>[nvtxs];
	std::cout << "NUM EDGES: " << belHeader.numEdges << std::endl;
	
    unsigned from, to; unsigned count;

	std::cerr << "READING GRAPH FROM FILE" << std::endl;
    while(infile.read((char*) &from, belHeader.sizeOfVertexType)) {
		infile.read((char*) &to, belHeader.sizeOfVertexType);
//        infile.read((char*) &count, belHeader.sizeOfCountType);

//        for(unsigned i=0; i<count; ++i) {
//            infile.read((char*) &to, belHeader.sizeOfVertexType);

//            edgeLists[from].insert(to);
//            edgeLists[to].insert(from);
//        }
    }

    infile.close();

    fprintf(stderr, "Creating CSR ...\n");

    idx_t numEdges = 0;
    for(idx_t i=0; i<nvtxs; ++i)
        numEdges += edgeLists[i].size();  

    idx_t ncon = 1;

    idx_t* xadj = new idx_t[nvtxs + 1];
    idx_t* adjncy = new idx_t[numEdges];
    idx_t adj_i = 0;
    for(idx_t i=0; i<nvtxs; ++i) {
        xadj[i] = adj_i;
        std::set<idx_t>::iterator it = edgeLists[i].begin();
        while(it != edgeLists[i].end()) {
            adjncy[adj_i++] = *it;
            ++it;    
        }
    }
    xadj[nvtxs] = adj_i;

    idx_t* vsize = new idx_t[nvtxs];
    idx_t* vwgt = new idx_t[nvtxs];
    for(idx_t i=0; i<nvtxs; ++i) {
        vsize[i] = 1;
        vwgt[i] = 1;
    }

    idx_t objval;
    idx_t* parts = new idx_t[nvtxs];

    fprintf(stderr, "Partitioning ...\n");

    METIS_PartGraphKway(&nvtxs, &ncon, xadj, adjncy, vwgt, vsize, NULL, &nparts, NULL, NULL, NULL, &objval, parts);
    
    fprintf(stderr, "Writing partitioning results ...\n");

    std::string commPath = partsDir + graphName + COMM_EXT;
    std::ofstream commFile;
    commFile.open(commPath.c_str());
    commFile << "Communication cost: " << objval << std::endl; 
    commFile.close();

    std::string partPath = partsDir + graphName + PART_EXT;
    std::ofstream partFile;
    partFile.open(partPath.c_str());
    for(idx_t i=0; i < nvtxs; ++i)
        partFile << parts[i] << std::endl;
    partFile.close();

    delete[] xadj;
    delete[] adjncy;
    delete[] vsize;
    delete[] vwgt;
    delete[] edgeLists;
    delete[] parts;

    return 0;
}
