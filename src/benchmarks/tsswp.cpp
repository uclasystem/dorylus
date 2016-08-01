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

//Lock mLock; 
//unsigned maximumLevel = 0;

typedef unsigned EType;

struct VType {
  unsigned value;
  IdType level;

  VType(unsigned v = 0, IdType l = VERY_HIGH) { value = v; level = l; }
};

IdType source;
std::set<IdType> checkers;

/* Basic version with no tagging */
template<typename VertexType, typename EdgeType>
class InitProgram : public VertexProgram<VertexType, EdgeType> {
  public:
    bool update(Vertex<VertexType, EdgeType>& vertex, EngineContext& engineContext) {
      bool changed = ((vertex.data().value != 0) || (vertex.data().level != VERY_HIGH));
      VType v(0, VERY_HIGH);
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
          if(sourceVertexData.level == VERY_HIGH) {
            //assert(sourceVertexData.parent == VERY_HIGH);
          } else
            orphan = false;

          break;
        }
      }

      if(orphan) {
        bool changed = ((vertex.data().value != 0) || (vertex.data().level != VERY_HIGH));
        VType v(0, VERY_HIGH);
        Engine<VertexType, EdgeType>::trimmed(vertex.globalId());
        vertex.setData(v);
        vertex.setParent(VERY_HIGH);
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
        assert((vertex.data().value == MAXWIDTH) && (vertex.data().level == 0));
        return false;
      }

      IdType myParent = vertex.parent(); 
      if(myParent == VERY_HIGH) {
        assert(vertex.data().level == VERY_HIGH);
        Engine<VertexType, EdgeType>::trimmed(vertex.globalId());
        Engine<VertexType, EdgeType>::shadowSignalVertex(vertex.globalId());
        return false;
      }

      bool orphan = true, reroot = false; VType candidate(0, VERY_HIGH); IdType cParent = VERY_HIGH;
      for(unsigned i=0; i<vertex.numInEdges(); ++i) {
        VertexType sourceVertexData = vertex.getSourceVertexData(i);
        if(vertex.getSourceVertexGlobalId(i) == myParent) {
          if(sourceVertexData.level == VERY_HIGH) {
            orphan = false; reroot = true;
            continue;
          } else {
            orphan = false; reroot = false;
            break;
          }
        }

        unsigned mW = std::min(sourceVertexData.value, vertex.getInEdgeData(i));
        if(mW > candidate.value) {
          if((mW > vertex.data().value) || ((mW == vertex.data().value) && (sourceVertexData.level < vertex.data().level))) {
            candidate.value = mW;
            cParent = vertex.getSourceVertexGlobalId(i);
            candidate.level = sourceVertexData.level + 1;
          }
        }
      }

      if(orphan || reroot) {
        //bool changed = ((vertex.data().value != candidate.value) || (vertex.data().level != candidate.level));
        bool changed = (cParent == VERY_HIGH); // We don't want to propagate other changes

        if(candidate.level == VERY_HIGH)
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
      if(vertex.globalId() == source) {
        assert((vertex.data().value == MAXWIDTH) && (vertex.data().level == 0));
        return false;
      }

      IdType myParent = vertex.parent();
      if(myParent == VERY_HIGH) {
        assert(vertex.data().level == VERY_HIGH);
        Engine<VertexType, EdgeType>::trimmed(vertex.globalId());
        Engine<VertexType, EdgeType>::shadowSignalVertex(vertex.globalId());
        return false;
      }

      if(checkers.find(vertex.globalId()) != checkers.end()) fprintf(stderr, "ASmartestTrimProgram: Processing vertex %u (level %u and parent %u) with %u in-edges\n", vertex.globalId(), vertex.data().level, vertex.parent(), vertex.numInEdges());

      bool orphan = true, reroot = false; VType candidate(0, VERY_HIGH); IdType cParent = VERY_HIGH;
      for(unsigned i=0; i<vertex.numInEdges(); ++i) {
        VertexType sourceVertexData = vertex.getSourceVertexData(i);
        if(checkers.find(vertex.globalId()) != checkers.end()) fprintf(stderr, "ASmartestTrimProgram: Vertex %u (level %u) source %u (level %u) has min(%u, %u) path\n", vertex.globalId(), vertex.data().level, vertex.getSourceVertexGlobalId(i), sourceVertexData.level, sourceVertexData.value, vertex.getInEdgeData(i));

        if(vertex.getSourceVertexGlobalId(i) == myParent) {
          //if((sourceVertexData.level >= vertex.data().level) && (sourceVertexData.level != VERY_HIGH)) 
            //fprintf(stderr, "SOMETHING IS WRONG: Vertex %u with <path, level> <%u, %u> has source %u with <path, level> <%u, %u>\n", vertex.globalId(), vertex.data().value, vertex.data().level, vertex.getSourceVertexGlobalId(i), sourceVertexData.value, sourceVertexData.level);
          //assert((sourceVertexData.level < vertex.data().level) || (sourceVertexData.level == VERY_HIGH));
          if((std::min(sourceVertexData.value, vertex.getInEdgeData(i)) < vertex.data().value) || (sourceVertexData.level >= vertex.data().level)) {
            orphan = false; reroot = true;
            //continue;
          } else {
            orphan = false; reroot = false;
            break;
          }
        }

        unsigned mW = std::min(sourceVertexData.value, vertex.getInEdgeData(i));
        if(mW > candidate.value) {
          if(sourceVertexData.level < vertex.data().level) {
            candidate.value = mW;
            cParent = vertex.getSourceVertexGlobalId(i);
            candidate.level = sourceVertexData.level + 1;
          }
        }
      }

      if(orphan || reroot) {
        //bool changed = ((vertex.data().value != candidate.value) || (vertex.data().level != candidate.level));
        bool changed = ((cParent == VERY_HIGH) || (candidate.value < vertex.data().value)); // We don't want to propagate other changes

        if(candidate.level == VERY_HIGH)
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


/* Approx version with tagging -- we look at the algorithm to find any possible viable path */
template<typename VertexType, typename EdgeType>
class ASmarterTrimProgram : public VertexProgram<VertexType, EdgeType> {
  public:
    bool update(Vertex<VertexType, EdgeType>& vertex, EngineContext& engineContext) {
      if(vertex.globalId() == source) { 
        assert((vertex.data().value == MAXWIDTH) && (vertex.data().level == 0));
        return false;
      }

      IdType myParent = vertex.parent();
      if(myParent == VERY_HIGH) {
        assert(vertex.data().level == VERY_HIGH);
        Engine<VertexType, EdgeType>::trimmed(vertex.globalId());
        Engine<VertexType, EdgeType>::shadowSignalVertex(vertex.globalId());
        return false;
      }

      if(checkers.find(vertex.globalId()) != checkers.end()) fprintf(stderr, "ASmarterTrimProgram: Processing vertex %u with %u in-edges\n", vertex.globalId(), vertex.numInEdges());

      unsigned maxWidth = 0; IdType maxParent = VERY_HIGH; IdType maxLevel = VERY_HIGH; 
      for(unsigned i=0; i<vertex.numInEdges(); ++i) {
        VertexType sourceVertexData = vertex.getSourceVertexData(i);
        if(checkers.find(vertex.globalId()) != checkers.end()) fprintf(stderr, "ASmarterTrimProgram: Vertex %u source %u has min(%u, %u) path\n", vertex.globalId(), vertex.getSourceVertexGlobalId(i), sourceVertexData.value, vertex.getInEdgeData(i));

        if(sourceVertexData.level < vertex.data().level) {
          unsigned mW = std::min(sourceVertexData.value, vertex.getInEdgeData(i));
          if(mW > maxWidth) {
            maxWidth = mW;
            maxParent = vertex.getSourceVertexGlobalId(i);
            maxLevel = sourceVertexData.level + 1;
          }
        }
      }

      if(checkers.find(vertex.globalId()) != checkers.end()) fprintf(stderr, "ASmarterTrimProgram: Vertex %u path %u maxParent = %u maxLevel = %u\n", vertex.globalId(), maxWidth, maxParent, maxLevel);

      //if((maxLevel != VERY_HIGH) && (maxLevel > vertex.data().level + 1))
      //fprintf(stderr, "DANGER: Vertex %u maxLevel = %u, VERY_HIGH = %u, vertex.data().level + 1 = %u\n", vertex.globalId(), maxLevel, VERY_HIGH, vertex.data().level + 1);

      assert((maxLevel == VERY_HIGH) || (maxLevel <= vertex.data().level));

/*
      if((maxLevel < VERY_HIGH) && (maxLevel > 4 * maximumLevel)) {
        fprintf(stderr, "Vertex %u is jumping all over (maximumLevel = %u); hence trimming it off\n", vertex.globalId(), maximumLevel);
        maxWidth = 0; maxParent = VERY_HIGH; maxLevel = VERY_HIGH;
      }
*/

      if((vertex.data().value != maxWidth) || (vertex.data().level != maxLevel)) {
        Engine<VertexType, EdgeType>::shadowSignalVertex(vertex.globalId());

        if(maxLevel == VERY_HIGH)
          Engine<VertexType, EdgeType>::trimmed(vertex.globalId());
        else
          Engine<VertexType, EdgeType>::notTrimmed(vertex.globalId());

        VType v(maxWidth, maxLevel);
        vertex.setData(v);
        vertex.setParent(maxParent);
        return true;
      }

      return false;
    }
};



/* Approx version with tagging */
// This is a monotonic function -- once the parent is attached in the dependence tree, it cannot change for the same value
template<typename VertexType, typename EdgeType>
class ASSWPProgram : public VertexProgram<VertexType, EdgeType> {
  public:
    bool update(Vertex<VertexType, EdgeType>& vertex, EngineContext& engineContext) {
      if(vertex.globalId() == source) { 
        bool changed = ((vertex.data().value != MAXWIDTH) || (vertex.data().level != 0));
        VType v(MAXWIDTH, 0);
        vertex.setData(v);
        vertex.setParent(VERY_HIGH);
        return changed;
      }

      if(checkers.find(vertex.globalId()) != checkers.end()) fprintf(stderr, "Processing vertex %u level %u with %u in-edges\n", vertex.globalId(), vertex.data().level, vertex.numInEdges());

      unsigned maxWidth = 0; IdType maxParent = VERY_HIGH; IdType maxLevel = VERY_HIGH; 
      for(unsigned i=0; i<vertex.numInEdges(); ++i) {
        VertexType sourceVertexData = vertex.getSourceVertexData(i);
        if(checkers.find(vertex.globalId()) != checkers.end()) fprintf(stderr, "Vertex %u level %u source %u level %u has min(%u, %u) path\n", vertex.globalId(), vertex.data().level, vertex.getSourceVertexGlobalId(i), sourceVertexData.level, sourceVertexData.value, vertex.getInEdgeData(i));

        unsigned mW = std::min(sourceVertexData.value, vertex.getInEdgeData(i));
        if(mW > maxWidth) {
          maxWidth = mW;
          maxParent = vertex.getSourceVertexGlobalId(i);
          maxLevel = sourceVertexData.level + 1;
        } /*else if((mW == maxWidth) && (sourceVertexData.level + 1 < maxLevel)) {
          maxParent = vertex.getSourceVertexGlobalId(i);
          maxLevel = sourceVertexData.level + 1;
        }*/
      }

      //if(checkers.find(vertex.globalId()) != checkers.end()) fprintf(stderr, "Vertex %u path %u level %u and parent %u\n", vertex.globalId(), maxWidth, maxLevel, maxParent);

      //if((vertex.data().value < maxWidth) || ((vertex.data().value == maxWidth) && (vertex.data().level != maxLevel))) {
      //if((vertex.data().value < maxWidth) || ((vertex.data().value == maxWidth) && (vertex.data().level > maxLevel))) {
      if(vertex.data().value < maxWidth) {  // We only need this condition -- effectively, once the parent is attached it cannot be changed
        if(checkers.find(vertex.globalId()) != checkers.end()) fprintf(stderr, "Vertex %u path %u level %u and parent %u\n", vertex.globalId(), maxWidth, maxLevel, maxParent);
        VType v(maxWidth, maxLevel);
        vertex.setData(v);
        vertex.setParent(maxParent);

/*
        mLock.lock();
        maximumLevel = std::max(maxLevel, maximumLevel);
        mLock.unlock();
*/
        return true;
      }

      //vertex.setParent(maxParent);

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

/* Approx version with tagging */
template<typename VertexType, typename EdgeType>
class FakeWriterProgram : public VertexProgram<VertexType, EdgeType> {
  public:

  void beforeIteration(EngineContext& engineContext) {
  }

  void processVertex(Vertex<VertexType, EdgeType>& vertex) {
  }

  void afterIteration(EngineContext& engineContext) {
  }
};

void ignoreAddDelete(VType& v) { }

EType edgeWeight(IdType from, IdType to) {
  return EType((from + to) % 255 + 1);
}

int main(int argc, char* argv[]) {
  init();
  //checkers.insert(0);
  //checkers.insert(93);

  //mLock.init();

  assert(parse(&argc, argv, "--bm-source=", &source));
  assert(parse(&argc, argv, "--bm-tmpdir=", tmpDir));

  unsigned smartPropagation = 100;
  assert(parse(&argc, argv, "--bm-smartpropagation=", &smartPropagation));
  assert(smartPropagation < 3);

  fprintf(stderr, "source = %u\n", source);
  fprintf(stderr, "tmpDir = %s\n", tmpDir);
  fprintf(stderr, "smartPropagation = %u\n", smartPropagation);

  VType defaultVertex = VType(0, VERY_HIGH);
  EType defaultEdge = 1;
  Engine<VType, EType>::init(argc, argv, defaultVertex, defaultEdge, &edgeWeight);

  Engine<VType, EType>::setOnAddDelete(DST, &ignoreAddDelete, DST, &ignoreAddDelete);
  Engine<VType, EType>::setOnDeleteSmartHandler(NULL);

  Engine<VType, EType>::signalAll();
  InitProgram<VType, EType> initProgram;
  Engine<VType, EType>::quickRun(&initProgram, true);

  ASSWPProgram<VType, EType> asswpProgram;
  //AWriterProgram<VType, EType> awriterProgram;
  FakeWriterProgram<VType, EType> awriterProgram;

  Engine<VType, EType>::signalVertex(source);

  switch(smartPropagation) {
    case 0: {
              fprintf(stderr, "Blind Trimming\n");
              ABlindTrimProgram<VType, EType> abtProgram;
              Engine<VType, EType>::streamRun5(&asswpProgram, &abtProgram, &awriterProgram, true);
              break;
            }
    case 1: {
              fprintf(stderr, "Smart Trimming\n");
              ASmartTrimProgram<VType, EType> astProgram;
              Engine<VType, EType>::streamRun5(&asswpProgram, &astProgram, &awriterProgram, true);
              break;
            }
    case 2: {
              fprintf(stderr, "Smarter Trimming\n");
              ASmartestTrimProgram<VType, EType> astProgram;
              Engine<VType, EType>::streamRun5(&asswpProgram, &astProgram, &awriterProgram, true);
              break;
            }
    default:
            assert(false);
  }

  Engine<VType, EType>::destroy();
  return 0;
}
