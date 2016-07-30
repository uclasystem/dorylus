/*
   3. Approx -- Tagging (OR) on Addition and Deletion to target
   4. Approx -- Tagging (OR) on Deletion to target 
 */
#include <iostream>
#include "../engine/engine.hpp"
#include "approximator.hpp"
#include <fstream>
#include <set>

#define MAXWIDTH 255

using namespace std;

char tmpDir[256];
bool smartPropagation = false;

typedef unsigned EType;
typedef unsigned VType;

IdType source;
std::set<IdType> checkers;

/* Basic version with no tagging */
template<typename VertexType, typename EdgeType>
class InitProgram : public VertexProgram<VertexType, EdgeType> {
  public:
    bool update(Vertex<VertexType, EdgeType>& vertex, EngineContext& engineContext) {
      bool changed = (vertex.data() != 0);
      vertex.setData(0);
      return changed;
    }
};

/* Approx version with tagging */
template<typename VertexType, typename EdgeType>
class ApproxResetProgram : public VertexProgram<VertexType, EdgeType> {
  public:
    bool update(Vertex<VertexType, EdgeType>& vertex, EngineContext& engineContext) {
      if(vertex.globalId() == source) {
        //assert(vertex.data() == MAXWIDTH);  // this is always true
        return false;
      }

      if(approximator::isApprox(vertex.data())) {
        if(checkers.find(vertex.globalId()) != checkers.end()) fprintf(stderr, "Processing vertex %u by resetting to 0\n", vertex.globalId());
        vertex.setData(0);
        return true;
      }

      return false;
    }
};

/* Approx version with tagging */
template<typename VertexType, typename EdgeType>
class ASSWPProgram : public VertexProgram<VertexType, EdgeType> {
  public:
    bool update(Vertex<VertexType, EdgeType>& vertex, EngineContext& engineContext) {
      if(vertex.globalId() == source) { 
        bool changed = (vertex.data() != MAXWIDTH);  // this is not via approximator because source is always accurate
        vertex.setData(MAXWIDTH); // this is not via approximator because source is always accurate
        return changed;
      }

      if(checkers.find(vertex.globalId()) != checkers.end()) fprintf(stderr, "Processing vertex %u with %u in-edges\n", vertex.globalId(), vertex.numInEdges());

      VType maxWidth = 0; bool approx = (approximator::isApprox(vertex.data())); 
      for(unsigned i=0; i<vertex.numInEdges(); ++i) {
        if(checkers.find(vertex.globalId()) != checkers.end()) fprintf(stderr, "Vertex %u source %u has min(%u, %u) path and approx = %s\n", vertex.globalId(), vertex.getSourceVertexGlobalId(i), approximator::value(vertex.getSourceVertexData(i)), vertex.getInEdgeData(i), approximator::isApprox(vertex.getSourceVertexData(i)) ? "true" : "false");

        maxWidth = std::max(maxWidth, std::min(approximator::value(vertex.getSourceVertexData(i)), vertex.getInEdgeData(i)));
        approx |= approximator::isApprox(vertex.getSourceVertexData(i));
      }

      if(checkers.find(vertex.globalId()) != checkers.end()) fprintf(stderr, "Vertex %u path %u and approx %s\n", vertex.globalId(), maxWidth, approx ? "true" : "false");

      if(approx) {
        approximator::setApprox(maxWidth);
        Engine<VertexType, EdgeType>::shadowSignalVertex(vertex.globalId());
      }

      bool changed = (vertex.data() != maxWidth);

      //if(checkers.find(vertex.globalId()) != checkers.end()) fprintf(stderr, "Vertex %u old path =  %u has new path = %u\n", vertex.globalId(), vertex.data(), maxWidth);

      vertex.setData(maxWidth);
      return changed;
    }
};

/* Approx version with tagging */
template<typename VertexType, typename EdgeType>
class ATagProgram : public VertexProgram<VertexType, EdgeType> {
  public:
    bool update(Vertex<VertexType, EdgeType>& vertex, EngineContext& engineContext) {
      if(vertex.globalId() == source) { 
        bool changed = (vertex.data() != MAXWIDTH);  // this is not via approximator because source is always accurate
        vertex.setData(MAXWIDTH); // this is not via approximator because source is always accurate
        return changed;
      }

      if(checkers.find(vertex.globalId()) != checkers.end()) fprintf(stderr, "ATag: Processing vertex %u with %u in-edges\n", vertex.globalId(), vertex.numInEdges());

      bool approx = (approximator::isApprox(vertex.data()));
      for(unsigned i=0; i<vertex.numInEdges(); ++i) {
        if(checkers.find(vertex.globalId()) != checkers.end()) fprintf(stderr, "Vertex %u source %u has min(%u, %u) path and approx = %s\n", vertex.globalId(), vertex.getSourceVertexGlobalId(i), approximator::value(vertex.getSourceVertexData(i)), vertex.getInEdgeData(i), approximator::isApprox(vertex.getSourceVertexData(i)) ? "true" : "false");

        if(smartPropagation) {
          if(std::min(approximator::value(vertex.getSourceVertexData(i)), vertex.getInEdgeData(i)) == approximator::value(vertex.data()))
            approx |= approximator::isApprox(vertex.getSourceVertexData(i)); 
        } else
          approx |= approximator::isApprox(vertex.getSourceVertexData(i));
      }
      
      VType vData = vertex.data();
      if(approx) {
        approximator::setApprox(vData);
        Engine<VertexType, EdgeType>::shadowSignalVertex(vertex.globalId());
        Engine<VertexType, EdgeType>::trimmed(vertex.globalId());
      }

      bool changed = (vertex.data() != vData);

      //if(checkers.find(vertex.globalId()) != checkers.end()) fprintf(stderr, "Vertex %u old path =  %u has new path = %u\n", vertex.globalId(), vertex.data(), maxWidth);

      vertex.setData(vData);
      return changed;
    }
};

/* Approx version with tagging */
template<typename VertexType, typename EdgeType>
class AWriterProgram : public VertexProgram<VertexType, EdgeType> {
  std::ofstream outFile;
  std::ofstream approxFile;
  public:

  void beforeIteration(EngineContext& engineContext) {
    char filename[300];
    sprintf(filename, "%s/output_%u_%u", tmpDir, engineContext.currentBatch(), NodeManager::getNodeId());
    outFile.open(filename);

    char afilename[300];
    sprintf(afilename, "%s/approx_%u_%u", tmpDir, engineContext.currentBatch(), NodeManager::getNodeId());
    approxFile.open(afilename);
  }

  void processVertex(Vertex<VertexType, EdgeType>& vertex) {
    if(checkers.find(vertex.globalId()) != checkers.end())
      fprintf(stderr, "Writer: Vertex %u path = %u\n", vertex.globalId(), vertex.data());

    outFile << vertex.globalId() << " " << approximator::value(vertex.data()) << std::endl;
    approxFile << vertex.globalId() << " " << (approximator::isApprox(vertex.data()) ? "true" : "false") << std::endl;
    //if(smartPropagation) vertex.setOldData(approximator::value(vertex.data()));
  }

  void afterIteration(EngineContext& engineContext) {
    outFile.close();
    approxFile.close();
  }
};

void setApprox(VType& v) {
  approximator::setApprox(v);
}

void setSmartApprox(VType& v, LightEdge<VType, EType>& e) {
  assert(e.to == v);
  if(std::min(approximator::value(e.from), e.edge) == v)
    approximator::setApprox(v);
}

void ignoreApprox(VType& v) { }

EType edgeWeight(IdType from, IdType to) {
  return EType((from + to + 25) % 50 + 1);
}

int main(int argc, char* argv[]) {
  init();
  //checkers.insert(750447);
  //checkers.insert(2591);

  assert(parse(&argc, argv, "--bm-source=", &source));
  assert(parse(&argc, argv, "--bm-tmpdir=", tmpDir));

  int t;
  assert(parse(&argc, argv, "--bm-smartpropagation=", &t));
  smartPropagation = (t == 0) ? false : true;

  //assert(smartPropagation == true);

  fprintf(stderr, "source = %u\n", source);
  fprintf(stderr, "tmpDir = %s\n", tmpDir);
  fprintf(stderr, "smartPropagation = %s\n", smartPropagation ? "true" : "false");

  VType defaultVertex = 0;
  EType defaultEdge = 1;
  Engine<VType, EType>::init(argc, argv, defaultVertex, defaultEdge, &edgeWeight);

  Engine<VType, EType>::setOnAddDelete(DST, &ignoreApprox, DST, &setApprox);
  Engine<VType, EType>::setOnDeleteSmartHandler(&setSmartApprox);

  Engine<VType, EType>::signalAll();
  InitProgram<VType, EType> initProgram;
  Engine<VType, EType>::quickRun(&initProgram, true);

  ATagProgram<VType, EType> atagProgram;
  ASSWPProgram<VType, EType> asswpProgram;
  ApproxResetProgram<VType, EType> approxResetProgram;
  AWriterProgram<VType, EType> awriterProgram;

  Engine<VType, EType>::signalVertex(source);

  Engine<VType, EType>::streamRun3(&asswpProgram, &atagProgram, &approxResetProgram, &asswpProgram, &awriterProgram, true, true);

  Engine<VType, EType>::destroy();
  return 0;
}
