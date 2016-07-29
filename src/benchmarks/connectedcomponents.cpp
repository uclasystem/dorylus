/*
   1. Baseline with reset which produces exact results
   2. Baseline without reset which produces unknown approximate results
 */

#include <iostream>
#include "../engine/engine.hpp"
#include <fstream>
#include <set>

#define MAXWIDTH 255

using namespace std;

char tmpDir[256];

typedef Empty EType;
typedef IdType VType;

std::set<IdType> checkers;

template<typename VertexType, typename EdgeType>
class InitProgram : public VertexProgram<VertexType, EdgeType> {
  public:
    bool update(Vertex<VertexType, EdgeType>& vertex, EngineContext& engineContext) {
      assert(vertex.numInEdges() == vertex.numOutEdges());
      bool changed = (vertex.data() != vertex.globalId());
      if(checkers.find(vertex.globalId()) != checkers.end()) fprintf(stderr, "InitProgram: Vertex %u has value %u and hence, changed = %s\n", vertex.globalId(), vertex.data(), changed ? "true" : "false"); 
      vertex.setData(vertex.globalId());
      return changed;
    }
};

template<typename VertexType, typename EdgeType>
class ConnectedComponentsProgram : public VertexProgram<VertexType, EdgeType> {
  public:
    bool update(Vertex<VertexType, EdgeType>& vertex, EngineContext& engineContext) {
      if(checkers.find(vertex.globalId()) != checkers.end()) fprintf(stderr, "Processing vertex %u with %u in-edges\n", vertex.globalId(), vertex.numInEdges());

      VType minComponent = vertex.data();
      for(unsigned i=0; i<vertex.numInEdges(); ++i) {
        if(checkers.find(vertex.globalId()) != checkers.end()) fprintf(stderr, "Vertex %u source %u has %u component\n", vertex.globalId(), vertex.getSourceVertexGlobalId(i), vertex.getSourceVertexData(i));

        minComponent = std::min(minComponent, vertex.getSourceVertexData(i));
      }
      bool changed = (vertex.data() != minComponent);

      if(checkers.find(vertex.globalId()) != checkers.end()) fprintf(stderr, "Vertex %u old component =  %u has new component = %u\n", vertex.globalId(), vertex.data(), minComponent);

      vertex.setData(minComponent);
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
      fprintf(stderr, "Vertex %u component = %u\n", vertex.globalId(), vertex.data());

    outFile << vertex.globalId() << " " << vertex.data() << std::endl; 
  }

  void afterIteration(EngineContext& engineContext) {
    outFile.close();
  }
};

EType edgeWeight(IdType from, IdType to) {
  return Empty();
}

void resetComputations() {
  Engine<VType, EType>::signalAll();
  InitProgram<VType, EType> initProgram;
  Engine<VType, EType>::quickRun(&initProgram, true);
  Engine<VType, EType>::signalAll();
}

void ignoreApprox(VType& v) { }

int main(int argc, char* argv[]) {
  init();
  //checkers.insert(43112);
  //checkers.insert(1965203);
  //checkers.insert(1965204);
  //checkers.insert(1965211);

  int rs;
  parse(&argc, argv, "--bm-reset=", &rs);
  bool reset = (rs == 0) ? false : true;

  parse(&argc, argv, "--bm-tmpdir=", tmpDir);

  fprintf(stderr, "reset = %s\n", reset ? "true" : "false");
  fprintf(stderr, "tmpDir = %s\n", tmpDir);

  VType defaultVertex = MAX_IDTYPE;
  EType defaultEdge = Empty();
  Engine<VType, EType>::init(argc, argv, defaultVertex, defaultEdge, &edgeWeight);
  Engine<VType, EType>::setOnAddDelete(DST, &ignoreApprox, DST, &ignoreApprox);

  Engine<VType, EType>::signalAll();
  InitProgram<VType, EType> initProgram;
  Engine<VType, EType>::quickRun(&initProgram, true);

  Engine<VType, EType>::signalAll();

  ConnectedComponentsProgram<VType, EType> connectedcomponentsProgram;
  WriterProgram<VType, EType> writerProgram;

  if(reset)
    Engine<VType, EType>::streamRun(&connectedcomponentsProgram, &writerProgram, &resetComputations, true);
  else
    Engine<VType, EType>::streamRun(&connectedcomponentsProgram, &writerProgram, NULL, true);

  Engine<VType, EType>::destroy();
  return 0;
}
