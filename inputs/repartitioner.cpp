#include <iostream>
#include <fstream>
#include <sstream>
#include <set>
#include <cassert>
#include <cstdlib>
#include <sys/stat.h>
#include <sys/types.h>

#include "/vagrant/install/metis-5.1.0/include/metis.h"

#define BASE_PATH "./"
#define PARTS_PATH BASE_PATH "parts_"
#define COMM_EXT ".comm"
#define PART_EXT ".parts"
#define SUBPART_EXT ".subparts"

int main(int argc, char* argv[])
{
    if(argc < 6) {
        fprintf(stderr, "NEVER INVOKE ME DIRECTLY! USE partitioner.sh SCRIPT WHICH WILL INTERNALLY INVOKE ME!\n");
        fprintf(stderr, "Dude! Invoke like this: <graph-file> <parts-file> <num-vertices> <original-partitions> <num-partitions>\n");
        exit(-1);
    }

    std::string graphName = argv[1];
    std::string partitionsFile = argv[2];
    idx_t nvtxs = atoll(argv[3]);
    idx_t ogparts = atoi(argv[4]);
    idx_t nparts = atoi(argv[5]);

    std::string partsDir = std::string(PARTS_PATH) + argv[4] + "/";
    mkdir(partsDir.c_str(), 0777);

    idx_t vertexToPartitions[nvtxs];
    idx_t globalToLocalId[nvtxs];
    idx_t numVertices[ogparts];
    for(idx_t i=0; i<ogparts; ++i)
        numVertices[i] = 0;

    {
        std::string partitionsFile = partsDir + graphName + PART_EXT;
        std::ifstream pfile(partitionsFile.c_str());
        std::string line; idx_t vct = 0;
        while(std::getline(pfile, line)) {
            if(line.size() == 0)
                continue;

            if(line[0] < '0' || line[0] > '9')
                assert(false);

            std::istringstream iss(line);
            if(!(iss >> vertexToPartitions[vct]))
                break;

            globalToLocalId[vct] = numVertices[vertexToPartitions[vct]];
            ++numVertices[vertexToPartitions[vct]];
            ++vct;
        }

        assert(vct == nvtxs);
    }


    fprintf(stderr, "Reading input graph ...\n");
    std::set<idx_t>** edgeLists = new std::set<idx_t>*[ogparts];
    for(idx_t i=0; i<ogparts; ++i)
        edgeLists[i] = new std::set<idx_t>[numVertices[i]];

    std::string graphPath = BASE_PATH + graphName;
    std::ifstream infile(graphPath.c_str());
    idx_t to, from;
    std::string line;
    while(std::getline(infile, line)) {
        if(line.size() == 0 || (line[0] < '0' || line[0] > '9'))
            continue;

        std::istringstream iss(line);
        if(!(iss >> from >> to))
            break;

        if(vertexToPartitions[from] == vertexToPartitions[to]) {
            idx_t partId = vertexToPartitions[from];
            edgeLists[partId][globalToLocalId[from]].insert(globalToLocalId[to]);
            edgeLists[partId][globalToLocalId[to]].insert(globalToLocalId[from]);
        }
    }

    idx_t** parts = new idx_t*[ogparts];
    for(idx_t i=0; i<ogparts; ++i)
        parts[i] = new idx_t[numVertices[i]];

    for(idx_t np=0; np<ogparts; ++np) {
        fprintf(stderr, "Creating CSR for subgraph %ld ...\n", np);

        idx_t numEdges = 0;
        for(idx_t i=0; i<numVertices[np]; ++i)
            numEdges += edgeLists[np][i].size();

        idx_t ncon = 1;

        idx_t* xadj = new idx_t[numVertices[np] + 1];
        idx_t* adjncy = new idx_t[numEdges];
        idx_t adj_i = 0;

        for(idx_t i=0; i<numVertices[np]; ++i) {
            xadj[i] = adj_i;
            std::set<idx_t>::iterator it = edgeLists[np][i].begin();
            while(it != edgeLists[np][i].end()) {
                adjncy[adj_i++] = *it;
                ++it;    
            }
        }
        xadj[numVertices[np]] = adj_i;

        idx_t* vsize = new idx_t[numVertices[np]];
        idx_t* vwgt = new idx_t[numVertices[np]];
        for(idx_t i=0; i<numVertices[np]; ++i) {
            vsize[i] = 1;
            vwgt[i] = 1;
        }

        idx_t objval;
        parts[np] = new idx_t[numVertices[np]];

        fprintf(stderr, "Partitioning %ld subgraph ...\n", np);

        METIS_PartGraphKway(&numVertices[np], &ncon, xadj, adjncy, vwgt, vsize, NULL, &nparts, NULL, NULL, NULL, &objval, parts[np]);
    
        fprintf(stderr, "Writing communication results ...\n");

        std::string commPath = partsDir + graphName + COMM_EXT;
        std::ofstream commFile;
        commFile.open(commPath.c_str(), std::ofstream::app);
        commFile << "Communication cost: " << objval << std::endl; 
        commFile.close();

        delete[] xadj;
        delete[] adjncy;
        delete[] vsize;
        delete[] vwgt;
    }

    fprintf(stderr, "Writing paritioning results ...\n");

    idx_t partsPtr[ogparts];
    for(idx_t i=0; i<ogparts; ++i)
        partsPtr[i] = 0;

    std::string partPath = partsDir + graphName + SUBPART_EXT;
    std::ofstream partFile;
    partFile.open(partPath.c_str());

    for(idx_t i=0; i < nvtxs; ++i) {
        idx_t partId = parts[vertexToPartitions[i]][partsPtr[vertexToPartitions[i]]++];
        if(partId >= vertexToPartitions[i])
            ++partId;
    
        partFile << partId << std::endl;
    }

    partFile.close();

    for(idx_t i=0; i<ogparts; ++i) {
        delete[] edgeLists[i];
        delete[] parts[i];
    }

    delete[] edgeLists;
    delete[] parts;

    return 0;
}
