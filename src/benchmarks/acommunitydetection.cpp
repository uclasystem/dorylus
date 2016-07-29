#include <iostream>
#include "../engine/densebitset.hpp"
#include "../engine/engine.hpp"
#include <fstream>
#include <map>

using namespace std;

typedef Empty EType;

typedef struct vType {
  unsigned community;
  bool approx;
  vType() { community = MAX_IDTYPE; approx = false; }
  vType(unsigned c, bool a) { community = c; approx = a; }
} VType;

DenseBitset approx;

template<typename VertexType, typename EdgeType>
class ApproxInit : public VertexProgram<VertexType, EdgeType> {
  public:
    bool update(Vertex<VertexType, EdgeType>& vertex, EngineContext& engineContext) {
      VType v(vertex.globalId(), approx.get(vertex.globalId()));

      bool changed = (v.approx != vertex.data().approx) | (v.community != vertex.data().community);
      if(changed)
        vertex.setData(v);
      return changed;
    }
};


template<typename VertexType, typename EdgeType>
class ACDProgram : public VertexProgram<VertexType, EdgeType> {
  public:
    bool update(Vertex<VertexType, EdgeType>& vertex, EngineContext& engineContext) {
      std::map<IdType, unsigned> frequencies;
      frequencies[vertex.data().community] = 1;
      unsigned maxCount = 1; IdType maxComp = vertex.data().community;
      bool vapprox = vertex.data().approx;

      for(unsigned i=0; i<vertex.numInEdges(); ++i) {
        IdType nc = vertex.getSourceVertexData(i).community;
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
        vapprox |= vertex.getSourceVertexData(i).approx;
      }

      bool changed = (vertex.data().community != maxComp) | (vertex.data().approx != vapprox);

      if(changed) {
        VType v(maxComp, vapprox);
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
    outFile << vertex.globalId() << " " << vertex.data().community << std::endl; 
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
    approx.setBit(v);
  }
}

int main(int argc, char* argv[]) {
  init();

  char approxFile[256];
  parse(&argc, argv, "--sssp-approxfile=", approxFile);
  fprintf(stderr, "ApproxFile = %s\n", approxFile);

  VType defaultVertex;
  Engine<VType, EType>::init(argc, argv, defaultVertex);

  approx.resize(Engine<VType, EType>::numVertices()); 
  approx.clear();

  loadApproxInfo(approxFile);

  if(Engine<VType, EType>::master()) fprintf(stderr, "Reduction approximates %u vertices (%.2f%)\n", approx.countSetBits(), (100.0 * approx.countSetBits()) / Engine<VType, EType>::numVertices());

  Engine<VType, EType>::signalAll();
  ApproxInit<VType, EType> approxInit;
  Engine<VType, EType>::run(&approxInit, false);

  Engine<VType, EType>::signalAll();
  ACDProgram<VType, EType> acdProgram;
  Engine<VType, EType>::run(&acdProgram, true);

  WriterProgram<VType, EType> writerProgram;
  Engine<VType, EType>::processAll(&writerProgram);

  Engine<VType, EType>::destroy();
  return 0;
}
