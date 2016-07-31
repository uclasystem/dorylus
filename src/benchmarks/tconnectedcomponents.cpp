/*
   5. Approx -- Tree construction
 */
#include <iostream>
#include "../engine/engine.hpp"
#include <fstream>
#include <set>

#define VERY_HIGH (MAX_IDTYPE - 10)

using namespace std;

char tmpDir[256];

Lock mLock; 
unsigned maximumLevel = 0;

typedef Empty EType;

struct VType {
  unsigned value;
  IdType level;

  VType(unsigned v = MAX_IDTYPE, IdType l = 0) { value = v; level = l; }
};

std::set<IdType> checkers;

/* Basic version with no tagging */
template<typename VertexType, typename EdgeType>
class InitProgram : public VertexProgram<VertexType, EdgeType> {
  public:
    bool update(Vertex<VertexType, EdgeType>& vertex, EngineContext& engineContext) {
      bool changed = ((vertex.data().value != vertex.globalId()) || (vertex.data().level != 0));
      if(checkers.find(vertex.globalId()) != checkers.end()) fprintf(stderr, "InitProgram: Vertex %u has value %u and hence, changed = %s\n", vertex.globalId(), vertex.data().value, changed ? "true" : "false");
      VType v(vertex.globalId(), 0);
      vertex.setData(v);
      vertex.setParent(VERY_HIGH);
      return changed;
    }
};

/* Approx version with blind tagging -- we don't look at anything related to algorithm here. This means, just trim the entire subtree */
template<typename VertexType, typename EdgeType>
class ABlindTrimProgram : public VertexProgram<VertexType, EdgeType> {
  public:
    bool update(Vertex<VertexType, EdgeType>& vertex, EngineContext& engineContext) {
      IdType myParent = vertex.parent(); 

      if(myParent == VERY_HIGH) {
        //assert(vertex.data().level == VERY_HIGH);
        Engine<VertexType, EdgeType>::trimmed(vertex.globalId());
        Engine<VertexType, EdgeType>::shadowSignalVertex(vertex.globalId());
        return false;
      }

      bool orphan = true;
      for(unsigned i=0; i<vertex.numInEdges(); ++i) {
        VertexType sourceVertexData = vertex.getSourceVertexData(i);
        if(vertex.getSourceVertexGlobalId(i) == myParent) {
          if(sourceVertexData.level == 0) {
            //assert(sourceVertexData.parent == VERY_HIGH);
          } else
            orphan = false;

          break;
        }
      }

      if(orphan) {
        bool changed = ((vertex.data().value != vertex.globalId()) || (vertex.data().level != 0));
        VType v(vertex.globalId(), 0);
        Engine<VertexType, EdgeType>::trimmed(vertex.globalId());
        vertex.setData(v);
        vertex.setParent(VERY_HIGH);
        Engine<VertexType, EdgeType>::shadowSignalVertex(vertex.globalId());
        return changed;
      }

      return false;
    }
};

/* Approx version with tagging -- we look at algorithm to find a better solution (if another component with lower value exists) */
template<typename VertexType, typename EdgeType>
class ASmartTrimProgram : public VertexProgram<VertexType, EdgeType> {
  public:
    bool update(Vertex<VertexType, EdgeType>& vertex, EngineContext& engineContext) {
      IdType myParent = vertex.parent(); 
      if(myParent == VERY_HIGH) {
        assert(vertex.data().level == 0);
        assert(vertex.data().value == vertex.globalId());
        return false;
      }

      bool orphan = true, reroot = false; VType candidate(vertex.globalId(), 0); IdType cParent = VERY_HIGH;
      for(unsigned i=0; i<vertex.numInEdges(); ++i) {
        VertexType sourceVertexData = vertex.getSourceVertexData(i);
        if(vertex.getSourceVertexGlobalId(i) == myParent) {
          if(sourceVertexData.level == 0) {
            orphan = false; reroot = true;
            continue;
          } else {
            orphan = false; reroot = false;
            break;
          }
        }

        unsigned mW = sourceVertexData.value;
        if(mW < candidate.value) {
          if((mW < vertex.data().value) || ((mW == vertex.data().value) && (sourceVertexData.level < vertex.data().level))) {
            candidate.value = mW;
            cParent = vertex.getSourceVertexGlobalId(i);
            candidate.level = sourceVertexData.level + 1;
          }
        }
      }

      if(orphan || reroot) {
        //bool changed = ((vertex.data().value != candidate.value) || (vertex.data().level != candidate.level));
        bool changed = (cParent == VERY_HIGH); // We don't want to propagate other changes

        if(candidate.level == 0)
          Engine<VertexType, EdgeType>::trimmed(vertex.globalId());
        else
          Engine<VertexType, EdgeType>::notTrimmed(vertex.globalId());

        VType v(candidate.value, candidate.level);
        vertex.setData(v);
        vertex.setParent(cParent);
        Engine<VertexType, EdgeType>::shadowSignalVertex(vertex.globalId());

        return changed;
      }

      return false;
    }
};

template<typename VertexType, typename EdgeType>
class ASmartestTrimProgram : public VertexProgram<VertexType, EdgeType> {
  public:
    bool update(Vertex<VertexType, EdgeType>& vertex, EngineContext& engineContext) {
      IdType myParent = vertex.parent();
      if(myParent == VERY_HIGH) {
        assert(vertex.data().level == 0);
        assert(vertex.data().value == vertex.globalId());
        return false;
      }

      if(checkers.find(vertex.globalId()) != checkers.end()) fprintf(stderr, "ASmartestTrimProgram: Processing vertex %u (level %u and parent %u) with %u in-edges\n", vertex.globalId(), vertex.data().level, vertex.parent(), vertex.numInEdges());

      bool orphan = true, reroot = false; VType candidate(vertex.globalId(), 0); IdType cParent = VERY_HIGH;
      for(unsigned i=0; i<vertex.numInEdges(); ++i) {
        VertexType sourceVertexData = vertex.getSourceVertexData(i);
        if(checkers.find(vertex.globalId()) != checkers.end()) fprintf(stderr, "ASmartestTrimProgram: Vertex %u (level %u) source %u (level %u) has %u path\n", vertex.globalId(), vertex.data().level, vertex.getSourceVertexGlobalId(i), sourceVertexData.level, sourceVertexData.value);

        if(vertex.getSourceVertexGlobalId(i) == myParent) {
          //if((sourceVertexData.level >= vertex.data().level) && (sourceVertexData.level != VERY_HIGH)) 
            //fprintf(stderr, "SOMETHING IS WRONG: Vertex %u with <path, level> <%u, %u> has source %u with <path, level> <%u, %u>\n", vertex.globalId(), vertex.data().value, vertex.data().level, vertex.getSourceVertexGlobalId(i), sourceVertexData.value, sourceVertexData.level);
          //assert((sourceVertexData.level < vertex.data().level) || (sourceVertexData.level == VERY_HIGH));
          if((sourceVertexData.value > vertex.data().value) || (sourceVertexData.level >= vertex.data().level)) {
            orphan = false; reroot = true;
            //continue;
          } else {
            orphan = false; reroot = false;
            break;
          }
        }

        unsigned mW = sourceVertexData.value;
        if(mW < candidate.value) {
          if(sourceVertexData.level < vertex.data().level) {
            candidate.value = mW;
            cParent = vertex.getSourceVertexGlobalId(i);
            candidate.level = sourceVertexData.level + 1;
          }
        }
      }

      if(orphan || reroot) {
        //bool changed = ((vertex.data().value != candidate.value) || (vertex.data().level != candidate.level));
        bool changed = ((cParent == VERY_HIGH) || (candidate.value > vertex.data().value)); // We don't want to propagate other changes

        if(candidate.level == 0)
          Engine<VertexType, EdgeType>::trimmed(vertex.globalId());
        else
          Engine<VertexType, EdgeType>::notTrimmed(vertex.globalId());

        if(checkers.find(vertex.globalId()) != checkers.end()) fprintf(stderr, "ASmartestTrimProgram: Vertex %u path %u maxParent = %u maxLevel = %u\n", vertex.globalId(), candidate.value, cParent, candidate.level);

        VType v(candidate.value, candidate.level);
        vertex.setData(v);
        vertex.setParent(cParent);
        Engine<VertexType, EdgeType>::shadowSignalVertex(vertex.globalId());

        return changed;
      }

      return false;
    }
};

/* Approx version with tagging */
template<typename VertexType, typename EdgeType>
class AConnectedComponentsProgram : public VertexProgram<VertexType, EdgeType> {
  public:
    bool update(Vertex<VertexType, EdgeType>& vertex, EngineContext& engineContext) {
      if(checkers.find(vertex.globalId()) != checkers.end()) fprintf(stderr, "Processing vertex %u with %u in-edges\n", vertex.globalId(), vertex.numInEdges());

      unsigned minComponent = vertex.data().value; IdType maxParent = vertex.parent(); IdType maxLevel = vertex.data().level; 
      for(unsigned i=0; i<vertex.numInEdges(); ++i) {
        VertexType sourceVertexData = vertex.getSourceVertexData(i);
        if(checkers.find(vertex.globalId()) != checkers.end()) fprintf(stderr, "Vertex %u source %u has %u component\n", vertex.globalId(), vertex.getSourceVertexGlobalId(i), sourceVertexData.value);

        unsigned mW = sourceVertexData.value;
        if(mW < minComponent) {
          minComponent = mW;
          maxParent = vertex.getSourceVertexGlobalId(i);
          maxLevel = sourceVertexData.level + 1;
        }
      }

      if(checkers.find(vertex.globalId()) != checkers.end()) fprintf(stderr, "Vertex %u component %u\n", vertex.globalId(), minComponent);

      if(vertex.data().value > minComponent) {
        VType v(minComponent, maxLevel);
        vertex.setData(v);
        vertex.setParent(maxParent);
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
  std::ofstream approxFile;
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
      fprintf(stderr, "Writer: Vertex %u component = %u parent = %u\n", vertex.globalId(), vertex.data().value, vertex.parent());

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
  assert(false);
  //v.value = MAX_IDTYPE;
  //v.parent = VERY_HIGH;
  //v.level = VERY_HIGH;
}

void setSmartTrimOff(VType& v, LightEdge<VType, EType>& e) {
  assert(false);
  /*
  if(v.parent == e.fromId) {
    fprintf(stderr, "Tree edge (%u -> %u) deleted\n", e.fromId, e.toId);
    v.value = e.toId;
    v.parent = VERY_HIGH;
    v.level = 0;
  }
  */
}

void setSmarterTrimOff(VType& v, LightEdge<VType, EType>& e) { }

EType edgeWeight(IdType from, IdType to) {
  return EType();
}

int main(int argc, char* argv[]) {
  init();
  //checkers.insert(43112);
  //checkers.insert(1965203);
  //checkers.insert(1965204);
  //checkers.insert(1965211);

  mLock.init();

  assert(parse(&argc, argv, "--bm-tmpdir=", tmpDir));

  unsigned smartPropagation = 100; 
  assert(parse(&argc, argv, "--bm-smartpropagation=", &smartPropagation));
  assert(smartPropagation < 3);

  fprintf(stderr, "tmpDir = %s\n", tmpDir);
  fprintf(stderr, "smartPropagation = %d\n", smartPropagation);

  VType defaultVertex = VType(MAX_IDTYPE, MAX_IDTYPE);
  EType defaultEdge = Empty();
  Engine<VType, EType>::init(argc, argv, defaultVertex, defaultEdge, &edgeWeight);

  Engine<VType, EType>::setOnAddDelete(DST, &ignoreApprox, DST, &ignoreApprox);
  Engine<VType, EType>::setOnDeleteSmartHandler(NULL);

  Engine<VType, EType>::signalAll();
  InitProgram<VType, EType> initProgram;
  Engine<VType, EType>::quickRun(&initProgram, true);

  AConnectedComponentsProgram<VType, EType> aconnectedcomponentsProgram;
  AWriterProgram<VType, EType> awriterProgram;

  Engine<VType, EType>::signalAll();

  switch(smartPropagation) {
    case 0: {
              fprintf(stderr, "Blind Trimming\n");
              ABlindTrimProgram<VType, EType> abtProgram;
              Engine<VType, EType>::streamRun5(&aconnectedcomponentsProgram, &abtProgram, &awriterProgram, true);
              break;
            }
    case 1: {
              fprintf(stderr, "Smart Trimming\n");
              ASmartTrimProgram<VType, EType> astProgram;
              Engine<VType, EType>::streamRun5(&aconnectedcomponentsProgram, &astProgram, &awriterProgram, true);
              break;
            }
    case 2: {
              fprintf(stderr, "Smarter Trimming\n");
              ASmartestTrimProgram<VType, EType> astProgram;
              Engine<VType, EType>::streamRun5(&aconnectedcomponentsProgram, &astProgram, &awriterProgram, true);
              break;
            }
    default:
            assert(false);
  }

  Engine<VType, EType>::destroy();
  return 0;
}
