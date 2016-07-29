#include <iostream>
#include "../engine/engine.hpp"
#include <fstream>
#include <string>
#include <sstream>
#include <set>

#define MAX_DIST 255

using namespace std;

typedef unsigned EType;
typedef unsigned VType;

std::set<IdType> checkers;
std::set<IdType> sources;

template<typename VertexType, typename EdgeType>
class SSSPProgram : public VertexProgram<VertexType, EdgeType> {
  public:
    bool update(Vertex<VertexType, EdgeType>& vertex, EngineContext& engineContext) {
      if(sources.find(vertex.globalId()) != sources.end()) {
        bool changed = (vertex.data() != 0);
        vertex.setData(0);
        return changed;
      }


      if(checkers.find(vertex.globalId()) != checkers.end()) fprintf(stderr, "Processing vertex %u with %u in-edges\n", vertex.globalId(), vertex.numInEdges());


      VType minPath = MAX_DIST;
      for(unsigned i=0; i<vertex.numInEdges(); ++i) {

        if(checkers.find(vertex.globalId()) != checkers.end()) fprintf(stderr, "Vertex %u source %u has %u path\n", vertex.globalId(), vertex.getSourceVertexGlobalId(i), vertex.getSourceVertexData(i));

        assert(vertex.getInEdgeData(i) == 1);
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
  WriterProgram() {
    char filename[10];
    sprintf(filename, "output_%u", NodeManager::getNodeId()); 
    outFile.open(filename);
  }

  void processVertex(Vertex<VertexType, EdgeType>& vertex) {
    if(checkers.find(vertex.globalId()) != checkers.end())
      fprintf(stderr, "Vertex %u path = %u\n", vertex.globalId(), vertex.data());

    outFile << vertex.globalId() << " " << vertex.data() << std::endl; 
  }

  ~WriterProgram() {
    outFile.close();
  }
};

int main(int argc, char* argv[]) {
  init();

  char csources[2048];
  parse(&argc, argv, "--mssp-sources=", csources);
  fprintf(stderr, "Sources = %s\n", csources);

  std::stringstream stream(csources);
  while(1) {
    IdType n;
    stream >> n;
    if(!stream) break;
    sources.insert(n);
    char c;
    stream >> c;
    if(!stream) break;
    assert(c == ',');
  } 

  fprintf(stderr, "sources: ");
  for(std::set<IdType>::iterator it = sources.begin(); it != sources.end(); ++it)
    fprintf(stderr, "%u ", *it);
  fprintf(stderr, "\n");

  VType defaultVertex = MAX_DIST;
  EType defaultEdge = 1;
  Engine<VType, EType>::init(argc, argv, defaultVertex, defaultEdge);

  for(std::set<IdType>::iterator it = sources.begin(); it != sources.end(); ++it)
    Engine<VType, EType>::signalVertex(*it);

  SSSPProgram<VType, EType> ssspProgram;
  Engine<VType, EType>::run(&ssspProgram, true);

  WriterProgram<VType, EType> writerProgram;
  Engine<VType, EType>::processAll(&writerProgram);

  Engine<VType, EType>::destroy();
  return 0;
}
