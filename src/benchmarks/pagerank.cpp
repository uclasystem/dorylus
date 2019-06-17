#include <iostream>
#include "../engine/engine.hpp"
#include <fstream>

using namespace std;

char tmpDir[256];

#define DAMPING_FACTOR 0.85
#define TOLERANCE 1.0e-2

typedef Empty EType;
typedef float VType;

template<typename VertexType, typename EdgeType>
class PageRankProgram : public VertexProgram<VertexType, EdgeType> {
public:
    bool update(Vertex<VertexType, EdgeType>& vertex, EngineContext& engineContext) {
        /*if(vertex.globalId() == 123)
            fprintf(stderr, "Processing vertex %u\n", vertex.globalId());
        */

        float sum = 0.0;
        for(unsigned i=0; i<vertex.numInEdges(); ++i) {
          /*  if(vertex.globalId() < 2)
                fprintf(stderr, "Vertex %u source %u has %.3lf rank\n", vertex.globalId(), vertex.getSourceVertexGlobalId(i), vertex.getSourceVertexData(i)); */
            sum += vertex.getSourceVertexData(i);
        }
        float pr = 1.0 - DAMPING_FACTOR + DAMPING_FACTOR * sum;
        float oldPr = vertex.data() * vertex.numOutEdges();
        bool changed = (fabs(oldPr - pr) > TOLERANCE);

        /*
        if(vertex.globalId() < 2)
            fprintf(stderr, "Vertex %u has %u outEdges\n", vertex.globalId(), vertex.numOutEdges());
        */
        
        if(vertex.numOutEdges() != 0)
            pr /= vertex.numOutEdges();
        
        /*
        if(vertex.globalId() < 2)
            fprintf(stderr, "Vertex %u old rank =  %.3lf has new rank = %.3lf\n", vertex.globalId(), vertex.data(), pr);
        */

        vertex.setData(pr);
        return changed;
    }
};

template<typename VertexType, typename EdgeType>
class WriterProgram : public VertexProgram<VertexType, EdgeType> {
    std::ofstream outFile;
public:
    WriterProgram() {
        char filename[200];
        sprintf(filename, "%s/output_%u", tmpDir, NodeManager::getNodeId()); 
        outFile.open(filename);
    }

    void processVertex(Vertex<VertexType, EdgeType>& vertex) {
       outFile << vertex.globalId() << " " << (int(vertex.data() * vertex.numOutEdges() * 100)) / 100.0 << std::endl; 
    }

    ~WriterProgram() {
        outFile.close();
    }
};

int main(int argc, char* argv[]) {
    init();

    parse(&argc, argv, "--bm-tmpdir=", tmpDir);

    VType defaultVertex = 1.0;
    Engine<VType, EType>::init(argc, argv, defaultVertex);
    Engine<VType, EType>::signalAll();
    
    PageRankProgram<VType, EType> pagerankProgram;
    Engine<VType, EType>::run(&pagerankProgram, true);

    WriterProgram<VType, EType> writerProgram;
    Engine<VType, EType>::processAll(&writerProgram);

    Engine<VType, EType>::destroy();
    return 0;
}
