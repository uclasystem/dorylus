/*
   5. Approx -- Tree construction
 */
#include <iostream>
#include "../engine/engine.hpp"
#include <fstream>
#include <set>

#define MAXWIDTH 255
#define VERY_HIGH (MAX_IDTYPE - 10)

using namespace std;

char tmpDir[256];
bool smartPropagation = false;

Lock mLock; 
unsigned maximumLevel = 0;

typedef unsigned EType;

struct VType {
  unsigned value;
  IdType parent;
  IdType level;

  VType(unsigned v = 0, IdType p = VERY_HIGH, IdType l = VERY_HIGH) { value = v; parent = p; level = l; }
};

IdType source;
std::set<IdType> checkers;

/* Basic version with no tagging */
template<typename VertexType, typename EdgeType>
class InitProgram : public VertexProgram<VertexType, EdgeType> {
  public:
    bool update(Vertex<VertexType, EdgeType>& vertex, EngineContext& engineContext) {
      bool changed = ((vertex.data().value != 0) || (vertex.data().parent != VERY_HIGH) || (vertex.data().level != VERY_HIGH));
      VType v(0, VERY_HIGH, VERY_HIGH);
      vertex.setData(v);
      return changed;
    }
};

/* Approx version with blind tagging -- we don't look at anything related to algorithm here. This means, just trim the entire subtree */
template<typename VertexType, typename EdgeType>
class ABlindTrimProgram : public VertexProgram<VertexType, EdgeType> {
  public:
    bool update(Vertex<VertexType, EdgeType>& vertex, EngineContext& engineContext) {
      if(vertex.globalId() == source) {
        bool changed = ((vertex.data().value != 0) || (vertex.data().parent != VERY_HIGH) || (vertex.data().level != VERY_HIGH));
        VType v(0, VERY_HIGH, VERY_HIGH);
        vertex.setData(v);
        return changed;
      }

      IdType myParent = vertex.data().parent; 

      if(myParent == VERY_HIGH) {
        assert(vertex.data().level == VERY_HIGH);
        Engine<VertexType, EdgeType>::trimmed(vertex.globalId());
        Engine<VertexType, EdgeType>::shadowSignalVertex(vertex.globalId());
        return false;
      }

      bool orphan = true;
      for(unsigned i=0; i<vertex.numInEdges(); ++i) {
        VertexType sourceVertexData = vertex.getSourceVertexData(i);
        if(vertex.getSourceVertexGlobalId(i) == myParent) {
          if(sourceVertexData.level == VERY_HIGH) {
            assert(sourceVertexData.parent == VERY_HIGH);
          } else
            orphan = false;

          break;
        }
      }

      if(orphan) {
        bool changed = ((vertex.data().value != 0) || (vertex.data().parent != VERY_HIGH) || (vertex.data().level != VERY_HIGH));
        VType v(0, VERY_HIGH, VERY_HIGH);
        Engine<VertexType, EdgeType>::trimmed(vertex.globalId());
        vertex.setData(v);
        Engine<VertexType, EdgeType>::shadowSignalVertex(vertex.globalId());
        return changed;
      }

      return false;
    }
};

/* Approx version with tagging -- we look at algorithm to find a better solution (if another path with higher value exists) */
template<typename VertexType, typename EdgeType>
class ASmartTrimProgram : public VertexProgram<VertexType, EdgeType> {
  public:
    bool update(Vertex<VertexType, EdgeType>& vertex, EngineContext& engineContext) {
      if(vertex.globalId() == source) {
        bool changed = ((vertex.data().value != MAXWIDTH) || (vertex.data().parent != VERY_HIGH) || (vertex.data().level != 0));
        VType v(MAXWIDTH, VERY_HIGH, 0);
        vertex.setData(v);
        return changed;
      }

      IdType myParent = vertex.data().parent; 

      if(myParent == VERY_HIGH) {
        assert(vertex.data().level == VERY_HIGH);
        Engine<VertexType, EdgeType>::trimmed(vertex.globalId());
        Engine<VertexType, EdgeType>::shadowSignalVertex(vertex.globalId());
        return false;
      }

      bool orphan = true, reroot = false; VType candidate(0, VERY_HIGH, VERY_HIGH);
      for(unsigned i=0; i<vertex.numInEdges(); ++i) {
        VertexType sourceVertexData = vertex.getSourceVertexData(i);
          if(vertex.getSourceVertexGlobalId(i) == myParent) {
            if(sourceVertexData.level == VERY_HIGH) {
              orphan = false; reroot = true;
              continue;
            } else 
              break;
        }

        unsigned mW = std::min(sourceVertexData.value, vertex.getInEdgeData(i));
        if(mW > candidate.value) {
          if((mW > vertex.data().value) || ((mW == vertex.data().value) && (sourceVertexData.level < vertex.data().level))) {
            candidate.value = mW;
            candidate.parent = vertex.getSourceVertexGlobalId(i);
            candidate.level = sourceVertexData.level + 1;
          }
        }
      }

      if(orphan || reroot) {
        bool changed = ((vertex.data().value != candidate.value) || (vertex.data().parent != candidate.parent) || (vertex.data().level != candidate.level));

        if(candidate.level == VERY_HIGH)
          Engine<VertexType, EdgeType>::trimmed(vertex.globalId());
        else
          Engine<VertexType, EdgeType>::notTrimmed(vertex.globalId());

        VType v(candidate.value, candidate.parent, candidate.level);
        vertex.setData(v);
        Engine<VertexType, EdgeType>::shadowSignalVertex(vertex.globalId());

        return changed;
      }

      return false;
    }
};

/* Approx version with tagging -- we look at the algorithm to find any possible viable path */
template<typename VertexType, typename EdgeType>
class ASmarterTrimProgram : public VertexProgram<VertexType, EdgeType> {
  public:
    bool update(Vertex<VertexType, EdgeType>& vertex, EngineContext& engineContext) {
      if(vertex.globalId() == source) { 
        bool changed = ((vertex.data().value != MAXWIDTH) || (vertex.data().parent != VERY_HIGH) || (vertex.data().level != 0));
        VType v(MAXWIDTH, VERY_HIGH, 0);
        vertex.setData(v);
        return changed;
      }

      if(engineContext.tooLong() || checkers.find(vertex.globalId()) != checkers.end()) fprintf(stderr, "ASmarterTrimProgram: Processing vertex %u with %u in-edges\n", vertex.globalId(), vertex.numInEdges());

      unsigned maxWidth = 0; IdType maxParent = VERY_HIGH; IdType maxLevel = VERY_HIGH; 
      for(unsigned i=0; i<vertex.numInEdges(); ++i) {
        VertexType sourceVertexData = vertex.getSourceVertexData(i);
        if(engineContext.tooLong() || checkers.find(vertex.globalId()) != checkers.end()) fprintf(stderr, "ASmarterTrimProgram: Vertex %u source %u has min(%u, %u) path\n", vertex.globalId(), vertex.getSourceVertexGlobalId(i), sourceVertexData.value, vertex.getInEdgeData(i));

        if(sourceVertexData.level < vertex.data().level) {
          unsigned mW = std::min(sourceVertexData.value, vertex.getInEdgeData(i));
          if(mW > maxWidth) {
            maxWidth = mW;
            maxParent = vertex.getSourceVertexGlobalId(i);
            maxLevel = sourceVertexData.level + 1;
          }
        }
      }

      if(engineContext.tooLong() || checkers.find(vertex.globalId()) != checkers.end()) fprintf(stderr, "ASmarterTrimProgram: Vertex %u path %u maxParent = %u maxLevel = %u\n", vertex.globalId(), maxWidth, maxParent, maxLevel);

      //if((maxLevel != VERY_HIGH) && (maxLevel > vertex.data().level + 1))
      //fprintf(stderr, "DANGER: Vertex %u maxLevel = %u, VERY_HIGH = %u, vertex.data().level + 1 = %u\n", vertex.globalId(), maxLevel, VERY_HIGH, vertex.data().level + 1);

      assert((maxLevel == VERY_HIGH) || (maxLevel <= vertex.data().level + 1));

      if((maxLevel < VERY_HIGH) && (maxLevel > 4 * maximumLevel)) {
        fprintf(stderr, "Vertex %u is jumping all over (maximumLevel = %u); hence trimming it off\n", vertex.globalId(), maximumLevel);
        maxWidth = 0; maxParent = VERY_HIGH; maxLevel = VERY_HIGH;
      }

      if((vertex.data().value != maxWidth) || (vertex.data().parent != maxParent) || (vertex.data().level != maxLevel)) {
        Engine<VertexType, EdgeType>::shadowSignalVertex(vertex.globalId());

        if(maxLevel == VERY_HIGH)
          Engine<VertexType, EdgeType>::trimmed(vertex.globalId());
        else
          Engine<VertexType, EdgeType>::notTrimmed(vertex.globalId());

        VType v(maxWidth, maxParent, maxLevel);
        vertex.setData(v);
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
        bool changed = ((vertex.data().value != MAXWIDTH) || (vertex.data().parent != VERY_HIGH) || (vertex.data().level != 0));
        VType v(MAXWIDTH, VERY_HIGH, 0);
        vertex.setData(v);
        return changed;
      }

      if(checkers.find(vertex.globalId()) != checkers.end()) fprintf(stderr, "Processing vertex %u with %u in-edges\n", vertex.globalId(), vertex.numInEdges());

      unsigned maxWidth = 0; IdType maxParent = VERY_HIGH; IdType maxLevel = VERY_HIGH; 
      for(unsigned i=0; i<vertex.numInEdges(); ++i) {
        VertexType sourceVertexData = vertex.getSourceVertexData(i);
        if(checkers.find(vertex.globalId()) != checkers.end()) fprintf(stderr, "Vertex %u source %u has min(%u, %u) path\n", vertex.globalId(), vertex.getSourceVertexGlobalId(i), sourceVertexData.value, vertex.getInEdgeData(i));

        unsigned mW = std::min(sourceVertexData.value, vertex.getInEdgeData(i));
        if(mW > maxWidth) {
          maxWidth = mW;
          maxParent = vertex.getSourceVertexGlobalId(i);
          maxLevel = sourceVertexData.level + 1;
        }
      }

      if(checkers.find(vertex.globalId()) != checkers.end()) fprintf(stderr, "Vertex %u path %u\n", vertex.globalId(), maxWidth);

      if(vertex.data().value < maxWidth) {
        VType v(maxWidth, maxParent, maxLevel);
        vertex.setData(v);
        mLock.lock();
        maximumLevel = std::max(maxLevel, maximumLevel);
        mLock.unlock();
        return true;
      }

      return false;
    }
};

/* Approx version with tagging */
template<typename VertexType, typename EdgeType>
class AWriterProgram : public VertexProgram<VertexType, EdgeType> {
  std::ofstream outFile;
  //std::ofstream approxFile;
  public:

  void beforeIteration(EngineContext& engineContext) {
    char filename[300];
    sprintf(filename, "%s/output_%u_%u", tmpDir, engineContext.currentBatch(), NodeManager::getNodeId());
    outFile.open(filename);

    //char afilename[300];
    //sprintf(afilename, "%s/approx_%u_%u", tmpDir, engineContext.currentBatch(), NodeManager::getNodeId());
    //approxFile.open(afilename);
  }

  void processVertex(Vertex<VertexType, EdgeType>& vertex) {
    if(checkers.find(vertex.globalId()) != checkers.end())
      fprintf(stderr, "Writer: Vertex %u path = %u\n", vertex.globalId(), vertex.data().value);

    outFile << vertex.globalId() << " " << vertex.data().value << std::endl;
    //approxFile << vertex.globalId() << " -> " << vertex.data().parent << std::endl;
  }

  void afterIteration(EngineContext& engineContext) {
    outFile.close();
    //approxFile.close();
  }
};

void setApprox(VType& v) {
  //approximator::setApprox(v);
  assert(false);
}

void setSmartApprox(VType& v, LightEdge<VType, EType>& e) {
  //assert(e.to == v);
  //if(std::min(approximator::value(e.from), e.edge) == v)
  //  approximator::setApprox(v);
  assert(false);
}

void ignoreApprox(VType& v) { }

void trimOff(VType& v) {
  v.value = 0;
  v.parent = VERY_HIGH;
  v.level = VERY_HIGH;
}

void setSmartTrimOff(VType& v, LightEdge<VType, EType>& e) {
  if(v.parent == e.fromId) {
    fprintf(stderr, "Tree edge (%u -> %u) deleted\n", e.fromId, e.toId);
    v.value = 0;
    v.parent = VERY_HIGH;
    v.level = VERY_HIGH;
  }
}

void setSmarterTrimOff(VType& v, LightEdge<VType, EType>& e) { }

EType edgeWeight(IdType from, IdType to) {
  return EType((from + to) % 255 + 1);
}

int main(int argc, char* argv[]) {
  init();
  //checkers.insert(3533649);
  //checkers.insert(2591);

  mLock.init();

  assert(parse(&argc, argv, "--bm-source=", &source));
  assert(parse(&argc, argv, "--bm-tmpdir=", tmpDir));

  int t;
  assert(parse(&argc, argv, "--bm-tagonadd=", &t));
  bool tagOnAdd = (t == 0) ? false : true;

  assert(parse(&argc, argv, "--bm-tagondelete=", &t));
  bool tagOnDelete = (t == 0) ? false : true;

  assert(parse(&argc, argv, "--bm-smarttagondelete=", &t));
  bool smartTagOnDelete = (t == 0) ? false : true;
  if(smartTagOnDelete)
    assert(tagOnAdd == false);

  assert(parse(&argc, argv, "--bm-smartpropagation=", &t));
  smartPropagation = (t == 0) ? false : true;

  fprintf(stderr, "source = %u\n", source);
  fprintf(stderr, "tmpDir = %s\n", tmpDir);
  fprintf(stderr, "tagOnAdd = %s\n", tagOnAdd ? "true" : "false");
  fprintf(stderr, "tagOnDelete = %s\n", tagOnDelete ? "true" : "false");
  fprintf(stderr, "smartTagOnDelete = %s\n", smartTagOnDelete ? "true" : "false");
  fprintf(stderr, "smartPropagation = %s\n", smartPropagation ? "true" : "false");

  VType defaultVertex = VType(0, VERY_HIGH, VERY_HIGH);
  EType defaultEdge = 1;
  Engine<VType, EType>::init(argc, argv, defaultVertex, defaultEdge, &edgeWeight);

  Engine<VType, EType>::setOnAddDelete(DST, (tagOnAdd ? &trimOff : &ignoreApprox), DST, (tagOnDelete ? &trimOff : &ignoreApprox));
  Engine<VType, EType>::setOnDeleteSmartHandler(smartTagOnDelete ? &setSmarterTrimOff : NULL);

  Engine<VType, EType>::signalAll();
  InitProgram<VType, EType> initProgram;
  Engine<VType, EType>::quickRun(&initProgram, true);

  ABlindTrimProgram<VType, EType> abtrimProgram;
  ASmartTrimProgram<VType, EType> astrimProgram;
  ASSWPProgram<VType, EType> asswpProgram;
  AWriterProgram<VType, EType> awriterProgram;

  Engine<VType, EType>::signalVertex(source);

  if(smartPropagation)
    Engine<VType, EType>::streamRun4(&asswpProgram, &astrimProgram, &awriterProgram, smartTagOnDelete, true);
  else
    Engine<VType, EType>::streamRun4(&asswpProgram, &abtrimProgram, &awriterProgram, smartTagOnDelete, true);

  Engine<VType, EType>::destroy();
  return 0;
}
