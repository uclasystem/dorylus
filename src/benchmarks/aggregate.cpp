#include <iostream>
#include "../engine/engine.hpp"
#include <fstream>
#include <memory>
#include <vector>
#include <cassert>
#include <iterator>

using namespace std;

char tmpDir[256];

typedef float EType;
typedef vector<FeatType> VType;

template<typename VertexType, typename EdgeType>
class AggregateProgram : public VertexProgram<VertexType, EdgeType> {
public:
    bool update(Vertex<VertexType, EdgeType>& vertex, EngineContext& engineContext) {
        VType curr = vertex.data();

        unsigned layer = engineContext.getIteration();

        for (unsigned i = 0; i < vertex.numInEdges(); ++i) {
            vector<FeatType> other = vertex.getSourceVertexDataAt(i, layer);
            sumVectors(curr, other);
            fprintf(stderr, "[[[]]] Vertex %u is adding vertex %u (layer %u, value %u)\n", vertex.globalId(), vertex.getInEdge(i).sourceId(), layer, other[1]);
        }

        vertex.addData(curr);   // Push to the back instead of modify the value.
        fprintf(stderr, "[[[]]] Vertex %u After: -> %u\n", vertex.globalId(), vertex.data()[1]);

        return false;
    }

private:
    void sumVectors(vector<FeatType>& curr, vector<FeatType>& other) {
        assert(curr.size() <= other.size());
        for (int i = 0; i < curr.size(); ++i) {
            curr[i] += other[i];
        }
    }
};


template<typename VertexType, typename EdgeType>
class WriterProgram : public VertexProgram<VertexType, EdgeType> {
    std::ofstream outFile;
public:
    WriterProgram() {
        char filename[200];
        sprintf(filename, "%s/output_%u", tmpDir, NodeManager::getNodeId()); 
        outFile.open(filename);
    }

    void processVertex(Vertex<VertexType, EdgeType>& vertex) {
        std::vector<VType>& data_all = vertex.dataAll();
        outFile << vertex.globalId() << ": ";
        for (int i = 0; i < data_all.size(); ++i) {
            VType curr = data_all[i];
            for (int j = 0; j < curr.size(); ++j) {
                outFile << curr[j] << " ";
            }
            outFile << "| ";
        }
        outFile << std::endl;
    }

    ~WriterProgram() {
        outFile.close();
    }
};

int main(int argc, char* argv[]) {
    init();

    parse(&argc, argv, "--bm-tmpdir=", tmpDir);

    VType defaultVertex = vector<FeatType>(2, 1);
    Engine<VType, EType>::init(argc, argv, defaultVertex);
    Engine<VType, EType>::signalAll();

    AggregateProgram<VType, EType> aggregateProgram;
    Engine<VType, EType>::run(&aggregateProgram, true);
    std::cerr << "finished Engine::run" << std::endl;

    WriterProgram<VType, EType> writerProgram;
    Engine<VType, EType>::processAll(&writerProgram);

    Engine<VType, EType>::destroy();
    return 0;
}
