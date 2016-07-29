/*
  1. Baseline with reset which produces exact results
  2. Baseline without reset which produces unknown approximate results
*/

#include <iostream>
#include "../engine/engine.hpp"
#include <fstream>
#include <set>

#define MAXPATH 255

using namespace std;

char tmpDir[256];

typedef unsigned EType;
typedef unsigned VType;

IdType source;
std::set<IdType> checkers;

template<typename VertexType, typename EdgeType>
class InitProgram : public VertexProgram<VertexType, EdgeType> {
  public:
    bool update(Vertex<VertexType, EdgeType>& vertex, EngineContext& engineContext) {
      bool changed = (vertex.data() != MAXPATH);
      vertex.setData(MAXPATH);
      return changed;
    }
};

template<typename VertexType, typename EdgeType>
class SSSPProgram : public VertexProgram<VertexType, EdgeType> {
  public:
    bool update(Vertex<VertexType, EdgeType>& vertex, EngineContext& engineContext) {
      if(vertex.globalId() == source) {
        bool changed = (vertex.data() != 0);
        vertex.setData(0);
        return changed;
      }

      if(checkers.find(vertex.globalId()) != checkers.end()) fprintf(stderr, "Processing vertex %u with %u in-edges\n", vertex.globalId(), vertex.numInEdges());

      VType minPath = MAXPATH;
      for(unsigned i=0; i<vertex.numInEdges(); ++i) {
        if(checkers.find(vertex.globalId()) != checkers.end()) fprintf(stderr, "Vertex %u source %u has min(%u, %u) path\n", vertex.globalId(), vertex.getSourceVertexGlobalId(i), vertex.getSourceVertexData(i), vertex.getInEdgeData(i));

        assert(vertex.getInEdgeData(i) > 0);
        minPath = std::min(minPath, vertex.getSourceVertexData(i) + vertex.getInEdgeData(i));
      }
      bool changed = (vertex.data() != minPath);

      if(checkers.find(vertex.globalId()) != checkers.end()) fprintf(stderr, "Vertex %u old path =  %u has new path = %u\n", vertex.globalId(), vertex.data(), minPath);

      vertex.setData(minPath);
      return changed;
    }
};

template<typename VertexType, typename EdgeType>
class WriterProgram : public VertexProgram<VertexType, EdgeType> {
  std::ofstream outFile;
  public:

  void beforeIteration(EngineContext& engineContext) {
    char filename[300];
    sprintf(filename, "%s/output_%u_%u", tmpDir, engineContext.currentBatch(), NodeManager::getNodeId());
    outFile.open(filename);
  }

  void processVertex(Vertex<VertexType, EdgeType>& vertex) {
    if(checkers.find(vertex.globalId()) != checkers.end())
      fprintf(stderr, "Vertex %u path = %u\n", vertex.globalId(), vertex.data());

    outFile << vertex.globalId() << " " << vertex.data() << std::endl; 
  }

  void afterIteration(EngineContext& engineContext) {
    outFile.close();
  }
};

EType edgeWeight(IdType from, IdType to) {
  return EType((from + to) % 255 + 1);
}

void ignoreApprox(VType& v) { }

int main(int argc, char* argv[]) {
  init();
  //checkers.insert(366);

  parse(&argc, argv, "--bm-source=", &source);
  
  int rs;
  parse(&argc, argv, "--bm-reset=", &rs);
  bool reset = (rs == 0) ? false : true;

  assert(reset == false);   // SSSP always produces right results

  parse(&argc, argv, "--bm-tmpdir=", tmpDir);

  fprintf(stderr, "source = %u\n", source);
  fprintf(stderr, "reset = %s\n", reset ? "true" : "false");
  fprintf(stderr, "tmpDir = %s\n", tmpDir);

  VType defaultVertex = MAXPATH;
  EType defaultEdge = 1;
  Engine<VType, EType>::init(argc, argv, defaultVertex, defaultEdge, &edgeWeight);
  Engine<VType, EType>::setOnAddDelete(DST, &ignoreApprox, DST, &ignoreApprox);

  Engine<VType, EType>::signalAll();
  InitProgram<VType, EType> initProgram;
  Engine<VType, EType>::quickRun(&initProgram, true);

  Engine<VType, EType>::signalVertex(source);

  SSSPProgram<VType, EType> ssspProgram;
  WriterProgram<VType, EType> writerProgram;

  Engine<VType, EType>::streamRun(&ssspProgram, &writerProgram, NULL, true);

  Engine<VType, EType>::destroy();
  return 0;
}
