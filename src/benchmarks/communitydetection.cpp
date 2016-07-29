#include <iostream>
#include "../engine/engine.hpp"
#include <fstream>
#include <map>

using namespace std;

typedef Empty EType;
typedef IdType VType;

template<typename VertexType, typename EdgeType>
class CDInit : public VertexProgram<VertexType, EdgeType> {
  public:
    bool update(Vertex<VertexType, EdgeType>& vertex, EngineContext& engineContext) {
      if(vertex.data() != vertex.globalId()) {
        vertex.setData(vertex.globalId());
        return true;
      }
      return false;
    }
};

template<typename VertexType, typename EdgeType>
class CDProgram : public VertexProgram<VertexType, EdgeType> {
  public:
    bool update(Vertex<VertexType, EdgeType>& vertex, EngineContext& engineContext) {
      std::map<IdType, unsigned> frequencies;
      frequencies[vertex.data()] = 1;
      unsigned maxCount = 1; IdType maxComp = vertex.data();

      for(unsigned i=0; i<vertex.numInEdges(); ++i) {
        //if(vertex.globalId() < 2) 
          //fprintf(stderr, "Vertex %u incoming edge from %u has community %u\n", vertex.globalId(), vertex.getSourceVertexGlobalId(i), vertex.getSourceVertexData(i));

        IdType nc = vertex.getSourceVertexData(i);
        std::map<IdType, unsigned>::iterator it = frequencies.find(nc);

        unsigned freq = 1;
        if(it != frequencies.end()) {
          it->second = it->second + 1;
          freq = it->second;
        }
        else
          frequencies[nc] = freq;

        if((freq > maxCount) || ((freq == maxCount) && (nc < maxComp))) {
          maxCount = freq;
          maxComp = nc;
        }
      }

      bool changed = (vertex.data() != maxComp);

      vertex.setData(maxComp);

      //if(vertex.globalId() < 2) 
        //fprintf(stderr, "Vertex %u community set to %u\n", vertex.globalId(), vertex.data());

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
    outFile << vertex.globalId() << " " << vertex.data() << std::endl; 
  }

  ~WriterProgram() {
    outFile.close();
  }
};

int main(int argc, char* argv[]) {
  init();

  VType defaultVertex = MAX_IDTYPE;
  Engine<VType, EType>::init(argc, argv, defaultVertex);

  Engine<VType, EType>::signalAll();
  CDInit<VType, EType> cdInit;
  Engine<VType, EType>::run(&cdInit, false); 

  Engine<VType, EType>::signalAll();
  CDProgram<VType, EType> cdProgram;
  Engine<VType, EType>::run(&cdProgram, true);

  WriterProgram<VType, EType> writerProgram;
  Engine<VType, EType>::processAll(&writerProgram);

  Engine<VType, EType>::destroy();
  return 0;
}
