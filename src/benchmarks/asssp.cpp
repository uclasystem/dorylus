#include <iostream>
#include "../engine/densebitset.hpp"
#include "../engine/engine.hpp"
#include <fstream>
#include <set>

#define MAX_DIST 255

using namespace std;

typedef unsigned EType;

typedef struct vType {
  unsigned path;
  bool approx;
  vType() { path = MAX_DIST; approx = false; }
  vType(unsigned p, bool a) { path = p; approx = a; }
} VType;

IdType source;
std::set<IdType> checkers;

DenseBitset approx;

template<typename VertexType, typename EdgeType>
class ApproxInit : public VertexProgram<VertexType, EdgeType> {
  public:
    bool update(Vertex<VertexType, EdgeType>& vertex, EngineContext& engineContext) {
      VType v;
      v.approx = approx.get(vertex.globalId());

      if(vertex.globalId() == source)
        v.path = 0;

      bool changed = (v.approx != vertex.data().approx) | (v.path != vertex.data().path);
      if(changed)
        vertex.setData(v);
      return changed;
    }
};


template<typename VertexType, typename EdgeType>
class ASSSPProgram : public VertexProgram<VertexType, EdgeType> {
    public:
        bool update(Vertex<VertexType, EdgeType>& vertex, EngineContext& engineContext) {
            /*
            if(vertex.globalId() == source) {
                bool changed = (vertex.data() != 0);
                vertex.setData(0);
                return changed;
            }
            */

            if(vertex.globalId() == source)
              return false;

            if(checkers.find(vertex.globalId()) != checkers.end()) fprintf(stderr, "Processing vertex %u with %u in-edges\n", vertex.globalId(), vertex.numInEdges());

            unsigned minPath = MAX_DIST; bool vapprox = vertex.data().approx;
            for(unsigned i=0; i<vertex.numInEdges(); ++i) {
                if(checkers.find(vertex.globalId()) != checkers.end()) fprintf(stderr, "Vertex %u source %u has %u path\n", vertex.globalId(), vertex.getSourceVertexGlobalId(i), vertex.getSourceVertexData(i).path);

                assert(vertex.getInEdgeData(i) == 1);
                minPath = std::min(minPath, vertex.getSourceVertexData(i).path + vertex.getInEdgeData(i));
                vapprox |= vertex.getSourceVertexData(i).approx;
            }

            bool changed = (vertex.data().path != minPath) | (vertex.data().approx != vapprox);

            if(checkers.find(vertex.globalId()) != checkers.end()) fprintf(stderr, "Vertex %u old path =  %u has new path = %u\n", vertex.globalId(), vertex.data().path, minPath);

            if(changed) {
              VType v(minPath, vapprox);
              vertex.setData(v);
            }

            return changed;
        }
};

template<typename VertexType, typename EdgeType>
class WriterProgram : public VertexProgram<VertexType, EdgeType> {
    std::ofstream outFile;
    std::ofstream approxOutFile;

    public:
    WriterProgram() {
        char filename[20];
        sprintf(filename, "output_%u", NodeManager::getNodeId()); 
        outFile.open(filename);

        sprintf(filename, "approx_%u", NodeManager::getNodeId());
        approxOutFile.open(filename);
    }

    void processVertex(Vertex<VertexType, EdgeType>& vertex) {
        if(checkers.find(vertex.globalId()) != checkers.end()) fprintf(stderr, "Vertex %u path = %u\n", vertex.globalId(), vertex.data().path);

        outFile << vertex.globalId() << " " << vertex.data().path << std::endl; 
        approxOutFile << vertex.globalId() << " " << (vertex.data().approx ? "true" : "false") << std::endl;
    }

    ~WriterProgram() {
        outFile.close();
        approxOutFile.close();
    }
};

void loadApproxInfo(char* approxFile) {
  std::ifstream infile(approxFile);
  if(!infile.good())
    fprintf(stderr, "Cannot open approx file: %s\n", approxFile);

  assert(infile.good());
  fprintf(stderr, "Reading approx file: %s\n", approxFile);

  IdType v; char o;

  while(infile >> v >> o) {
    if(o == 'A')
      approx.setBit(v);
  }
}

int main(int argc, char* argv[]) {
    init();
    //checkers.insert(917);

    parse(&argc, argv, "--sssp-source=", &source);
    fprintf(stderr, "Source = %u\n", source);

    char approxFile[256];
    parse(&argc, argv, "--sssp-approxfile=", approxFile);
    fprintf(stderr, "ApproxFile = %s\n", approxFile);

    VType defaultVertex;
    EType defaultEdge = 1;
    Engine<VType, EType>::init(argc, argv, defaultVertex, defaultEdge);

    approx.resize(Engine<VType, EType>::numVertices()); 
    approx.clear();

    loadApproxInfo(approxFile);

    if(Engine<VType, EType>::master()) fprintf(stderr, "Reduction approximates %u vertices (%.2f%)\n", approx.countSetBits(), (100.0 * approx.countSetBits()) / Engine<VType, EType>::numVertices());

    Engine<VType, EType>::signalAll();
    ApproxInit<VType, EType> approxInit;
    Engine<VType, EType>::run(&approxInit, false);

    Engine<VType, EType>::signalVertex(source);
    ASSSPProgram<VType, EType> assspProgram;
    Engine<VType, EType>::run(&assspProgram, true);

    WriterProgram<VType, EType> writerProgram;
    Engine<VType, EType>::processAll(&writerProgram);

    Engine<VType, EType>::destroy();
    return 0;
}
