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
bool smartPropagation = false;

Lock mLock; 
unsigned maximumLevel = 0;

typedef Empty EType;

struct VType {
  unsigned value;
  IdType parent;
  IdType level;

  VType(unsigned v = MAX_IDTYPE, IdType p = VERY_HIGH, IdType l = 0) { value = v; parent = p; level = l; }
};

std::set<IdType> checkers;

/* Basic version with no tagging */
template<typename VertexType, typename EdgeType>
class InitProgram : public VertexProgram<VertexType, EdgeType> {
  public:
    bool update(Vertex<VertexType, EdgeType>& vertex, EngineContext& engineContext) {
      bool changed = ((vertex.data().value != vertex.globalId()) || (vertex.data().parent != VERY_HIGH) || (vertex.data().level != 0));
      if(checkers.find(vertex.globalId()) != checkers.end()) fprintf(stderr, "InitProgram: Vertex %u has value %u and hence, changed = %s\n", vertex.globalId(), vertex.data().value, changed ? "true" : "false");
      VType v(vertex.globalId(), VERY_HIGH, 0);
      vertex.setData(v);
      return changed;
    }
};

/* Approx version with tagging */
template<typename VertexType, typename EdgeType>
class ABlindTrimProgram : public VertexProgram<VertexType, EdgeType> {
  public:
    bool update(Vertex<VertexType, EdgeType>& vertex, EngineContext& engineContext) {
      IdType myParent = vertex.data().parent; 

      if(myParent == VERY_HIGH) {
        //assert(vertex.data().level == VERY_HIGH);
        Engine<VertexType, EdgeType>::trimmed(vertex.globalId());
        Engine<VertexType, EdgeType>::shadowSignalVertex(vertex.globalId());

        if(vertex.data().value != vertex.globalId()) {
          VType v(vertex.globalId(), VERY_HIGH, 0);
          vertex.setData(v);
        }
        return false;
      }

      bool found = false; bool orphan = false;
      for(unsigned i=0; i<vertex.numInEdges(); ++i) {
        VertexType sourceVertexData = vertex.getSourceVertexData(i);
        if(vertex.getSourceVertexGlobalId(i) == myParent) {
          if((sourceVertexData.level == 0) && (vertex.data().value != sourceVertexData.value)) {
            assert(sourceVertexData.parent == VERY_HIGH);
            orphan = true;
          }

          found = true;
          break;
        }
      }

      assert(found);

      if(orphan) {
        bool changed = ((vertex.data().value != vertex.globalId()) || (vertex.data().parent != VERY_HIGH) || (vertex.data().level != 0));
        VType v(vertex.globalId(), VERY_HIGH, 0);
        Engine<VertexType, EdgeType>::trimmed(vertex.globalId());
        vertex.setData(v);
        Engine<VertexType, EdgeType>::shadowSignalVertex(vertex.globalId());
        return changed;
      }

      return false;
    }
};

/* Approx version with tagging */
template<typename VertexType, typename EdgeType>
class ASmartTrimProgram : public VertexProgram<VertexType, EdgeType> {
  public:
    bool update(Vertex<VertexType, EdgeType>& vertex, EngineContext& engineContext) {
      IdType myParent = vertex.data().parent; 

      if(myParent == VERY_HIGH) {
        assert(vertex.data().level == 0);
        //Engine<VertexType, EdgeType>::trimmed(vertex.globalId());
        //Engine<VertexType, EdgeType>::shadowSignalVertex(vertex.globalId());
        assert(vertex.data().value == vertex.globalId());
        //if(vertex.data().value != vertex.globalId()) {
        //  VType v(vertex.globalId(), VERY_HIGH, 0);
        //  vertex.setData(v);
        //}
        return false;
      }

      bool orphan = true; bool reroot = false; VType candidate(vertex.globalId(), VERY_HIGH, 0);
      for(unsigned i=0; i<vertex.numInEdges(); ++i) {
        VertexType sourceVertexData = vertex.getSourceVertexData(i);
        if(vertex.getSourceVertexGlobalId(i) == myParent) {
          if(checkers.find(vertex.globalId()) != checkers.end()) fprintf(stderr, "Vertex %u source %u has %u component\n", vertex.globalId(), vertex.getSourceVertexGlobalId(i), sourceVertexData.value);
          if(sourceVertexData.value != vertex.data().value) {
            assert(sourceVertexData.parent == VERY_HIGH);
            orphan = false; reroot = true;
            break;
          } else {
            if(sourceVertexData.level + 1 != vertex.data().level) {
              candidate.value = vertex.data().value;
              candidate.parent = vertex.data().parent;
              candidate.level = sourceVertexData.level + 1;
              reroot = true;
            }
            break;
          }
          //continue;
        }

        continue;

        unsigned mW = sourceVertexData.value;
        if(mW < candidate.value) {
          if((mW < vertex.data().value) || ((mW == vertex.data().value) && (sourceVertexData.level < vertex.data().level))) {
            candidate.value = mW;
            candidate.parent = vertex.getSourceVertexGlobalId(i);
            candidate.level = sourceVertexData.level + 1;
          }
        }
      }

      if(orphan || reroot) {
        bool changed = ((vertex.data().value != candidate.value) || (vertex.data().parent != candidate.parent) || (vertex.data().level != candidate.level));
        //bool changed = (candidate.parent == VERY_HIGH) && (vertex.data().parent != VERY_HIGH);

        if(candidate.parent == VERY_HIGH)
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

/* Approx version with tagging */
template<typename VertexType, typename EdgeType>
class ASmarterTrimProgram : public VertexProgram<VertexType, EdgeType> {
  public:
    bool update(Vertex<VertexType, EdgeType>& vertex, EngineContext& engineContext) {
      if(engineContext.tooLong() || checkers.find(vertex.globalId()) != checkers.end()) fprintf(stderr, "ASmarterTrimProgram: Processing vertex %u with %u in-edges\n", vertex.globalId(), vertex.numInEdges());

      /*
      if(vertex.data().level == 0) {
        assert(vertex.data().parent == VERY_HIGH);
        assert(vertex.data().value == vertex.globalId());
        return false;
      }
      */

      // EITHER COMMENT LIKE THIS, OR WE SHOULD SCHEDULE THE ABOVE VERTEX

      unsigned minComponent = vertex.globalId(); IdType maxParent = VERY_HIGH; IdType maxLevel = 0; 
      for(unsigned i=0; i<vertex.numInEdges(); ++i) {
        VertexType sourceVertexData = vertex.getSourceVertexData(i);
        if(engineContext.tooLong() || checkers.find(vertex.globalId()) != checkers.end()) fprintf(stderr, "ASmarterTrimProgram: Vertex %u source %u has %u component\n", vertex.globalId(), vertex.getSourceVertexGlobalId(i), sourceVertexData.value);

        if(sourceVertexData.level < vertex.data().level) {
          unsigned mW = sourceVertexData.value;
          if(mW < minComponent) {
            minComponent = mW;
            maxParent = vertex.getSourceVertexGlobalId(i);
            maxLevel = sourceVertexData.level + 1;
          }
        }
      }

      if(engineContext.tooLong() || checkers.find(vertex.globalId()) != checkers.end()) fprintf(stderr, "ASmarterTrimProgram: Vertex %u component %u maxParent = %u maxLevel = %u\n", vertex.globalId(), minComponent, maxParent, maxLevel);

      //if((maxLevel != VERY_HIGH) && (maxLevel > vertex.data().level + 1))
      //fprintf(stderr, "DANGER: Vertex %u maxLevel = %u, VERY_HIGH = %u, vertex.data().level + 1 = %u\n", vertex.globalId(), maxLevel, VERY_HIGH, vertex.data().level + 1);

      assert((maxLevel == VERY_HIGH) || (maxLevel <= vertex.data().level + 1));

      if((maxLevel < VERY_HIGH) && (maxLevel > 4 * maximumLevel)) {
        fprintf(stderr, "Vertex %u is jumping all over (maximumLevel = %u); hence trimming it off\n", vertex.globalId(), maximumLevel);
        minComponent = vertex.globalId(); maxParent = VERY_HIGH; maxLevel = 0;
      }

      if((vertex.data().value != minComponent) || (vertex.data().parent != maxParent) || (vertex.data().level != maxLevel)) {
        Engine<VertexType, EdgeType>::shadowSignalVertex(vertex.globalId());

        if(maxParent == VERY_HIGH)
          Engine<VertexType, EdgeType>::trimmed(vertex.globalId());
        else
          Engine<VertexType, EdgeType>::notTrimmed(vertex.globalId());

        VType v(minComponent, maxParent, maxLevel);
        vertex.setData(v);
        return true;
      }

      Engine<VertexType, EdgeType>::shadowSignalVertex(vertex.globalId());

      return false;
    }
};



/* Approx version with tagging */
template<typename VertexType, typename EdgeType>
class AConnectedComponentsProgram : public VertexProgram<VertexType, EdgeType> {
  public:
    bool update(Vertex<VertexType, EdgeType>& vertex, EngineContext& engineContext) {
      if(checkers.find(vertex.globalId()) != checkers.end()) fprintf(stderr, "Processing vertex %u with %u in-edges\n", vertex.globalId(), vertex.numInEdges());

      unsigned minComponent = vertex.data().value; IdType maxParent = vertex.data().parent; IdType maxLevel = vertex.data().level; 
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
        VType v(minComponent, maxParent, maxLevel);
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
      fprintf(stderr, "Writer: Vertex %u component = %u parent = %u\n", vertex.globalId(), vertex.data().value, vertex.data().parent);

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
  v.value = MAX_IDTYPE;
  v.parent = VERY_HIGH;
  v.level = VERY_HIGH;
}

void setSmartTrimOff(VType& v, LightEdge<VType, EType>& e) {
  if(v.parent == e.fromId) {
    fprintf(stderr, "Tree edge (%u -> %u) deleted\n", e.fromId, e.toId);
    v.value = e.toId;
    v.parent = VERY_HIGH;
    v.level = 0;
  }
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

  fprintf(stderr, "tmpDir = %s\n", tmpDir);
  fprintf(stderr, "tagOnAdd = %s\n", tagOnAdd ? "true" : "false");
  fprintf(stderr, "tagOnDelete = %s\n", tagOnDelete ? "true" : "false");
  fprintf(stderr, "smartTagOnDelete = %s\n", smartTagOnDelete ? "true" : "false");
  fprintf(stderr, "smartPropagation = %s\n", smartPropagation ? "true" : "false");

  VType defaultVertex = VType(MAX_IDTYPE, VERY_HIGH, 0);
  EType defaultEdge = Empty();
  Engine<VType, EType>::init(argc, argv, defaultVertex, defaultEdge, &edgeWeight);

  Engine<VType, EType>::setOnAddDelete(DST, (tagOnAdd ? &trimOff : &ignoreApprox), DST, (tagOnDelete ? &trimOff : &ignoreApprox));
  Engine<VType, EType>::setOnDeleteSmartHandler(smartTagOnDelete ? &setSmarterTrimOff : NULL);

  Engine<VType, EType>::signalAll();
  InitProgram<VType, EType> initProgram;
  Engine<VType, EType>::quickRun(&initProgram, true);

  ABlindTrimProgram<VType, EType> abtrimProgram;
  ASmarterTrimProgram<VType, EType> astrimProgram;
  AConnectedComponentsProgram<VType, EType> aconnectedcomponentsProgram;
  AWriterProgram<VType, EType> awriterProgram;

  Engine<VType, EType>::signalAll();

  if(smartPropagation)
    Engine<VType, EType>::streamRun4(&aconnectedcomponentsProgram, &astrimProgram, &awriterProgram, smartTagOnDelete, true);
  else
    Engine<VType, EType>::streamRun4(&aconnectedcomponentsProgram, &abtrimProgram, &awriterProgram, smartTagOnDelete, true);

  Engine<VType, EType>::destroy();
  return 0;
}
