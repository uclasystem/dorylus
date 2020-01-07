#include <fstream>
#include "graph.hpp"


/** For files cli options. */
#define RAWGRAPH_EXT "graph.bsnap"
#define EDGES_EXT ".edges"
#define PARTS_EXT ".parts"

/** Binary snap file header struct. */
struct BSHeaderType {
    int sizeOfVertexType;
    unsigned numVertices;
    unsigned long long numEdges;
};


class DataLoader {
public:
    DataLoader(std::string datasetDir, unsigned _nodeId, unsigned _numNodes, bool _undirected);
    ~DataLoader();

    void readPartsFile();
    void processEdge(unsigned &from, unsigned &to);
    void findGhostDegrees();
    void setEdgeNormalizations();
    void preprocess();

private:
    unsigned nodeId;
    unsigned numNodes;

    std::string graphFile;
    std::string partsFile;
    bool undirected;

    std::string processedGraphFile;

    RawGraph rawGraph;

    bool **forwardDstTables;
    bool **backwardDstTables;
};
