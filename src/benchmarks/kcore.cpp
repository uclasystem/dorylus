#include <iostream>
#include "../engine/engine.hpp"
#include <fstream>
#include <set>

#define MAX_CORE 500

using namespace std;

typedef unsigned EType;
typedef unsigned VType;

std::set<IdType> checkers;
unsigned maxCore;
unsigned core;

template<typename VertexType, typename EdgeType>
class KCoreProgram : public VertexProgram<VertexType, EdgeType> {
    public:
        bool update(Vertex<VertexType, EdgeType>& vertex, EngineContext& engineContext) {
            if(checkers.find(vertex.globalId()) != checkers.end()) fprintf(stderr, "Processing vertex %u with %u in-edges\n", vertex.globalId(), vertex.numInEdges());

            if(vertex.data() != MAX_CORE)
              return false;

            unsigned numLiveEdges = 0;
            for(unsigned i=0; i<vertex.numInEdges(); ++i) {
                if(checkers.find(vertex.globalId()) != checkers.end()) fprintf(stderr, "Vertex %u source %u has %u core\n", vertex.globalId(), vertex.getSourceVertexGlobalId(i), vertex.getSourceVertexData(i));

                if(vertex.getSourceVertexData(i) == MAX_CORE)
                  ++numLiveEdges;
            }

            if(numLiveEdges <= core) {
              vertex.setData(core);
              if(checkers.find(vertex.globalId()) != checkers.end()) fprintf(stderr, "Vertex %u core = %u\n", vertex.globalId(), vertex.data());
              return true;
            }

            return false;
        }
};

template<typename VertexType, typename EdgeType>
class WriterProgram : public VertexProgram<VertexType, EdgeType> {
    std::ofstream outFile;
    public:
    WriterProgram() {
        char filename[10];
        sprintf(filename, "output_%u", NodeManager::getNodeId()); 
        outFile.open(filename);
    }

    void processVertex(Vertex<VertexType, EdgeType>& vertex) {
        if(checkers.find(vertex.globalId()) != checkers.end())
            fprintf(stderr, "Vertex %u core = %u\n", vertex.globalId(), vertex.data());

        outFile << vertex.globalId() << " " << vertex.data() << std::endl; 
    }

    ~WriterProgram() {
        outFile.close();
    }
};

int main(int argc, char* argv[]) {
    init();

    parse(&argc, argv, "--kcore-maxcore=", &maxCore);
    fprintf(stderr, "maxCore = %u\n", maxCore);

    VType defaultVertex = MAX_CORE;
    EType defaultEdge = 1;
    Engine<VType, EType>::init(argc, argv, defaultVertex, defaultEdge);

    KCoreProgram<VType, EType> kcoreProgram;

    double tmr = -getTimer();
    for(core = 0; core < maxCore; ++core) {
      fprintf(stderr, "core = %u\n", core);
      Engine<VType, EType>::signalAll();
      Engine<VType, EType>::run(&kcoreProgram, true);
    } 
    fprintf(stderr, "KCore completed in %.3lf ms\n", tmr + getTimer());

    WriterProgram<VType, EType> writerProgram;
    Engine<VType, EType>::processAll(&writerProgram);

    Engine<VType, EType>::destroy();
    return 0;
}
