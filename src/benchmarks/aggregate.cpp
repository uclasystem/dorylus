#include <iostream>
#include "../engine/engine.hpp"
#include <fstream>
#include <vector>

using namespace std;

char tmpDir[256];

typedef Empty EType;
typedef unsigned VType;

template<typename VertexType, typename EdgeType>
class AggregateProgram : public VertexProgram<VertexType, EdgeType> {
public:
    bool update(Vertex<VertexType, EdgeType>& vertex, EngineContext& engineContext) {
	bool changed = true;
	VType curr = vertex.data();

	for (unsigned i = 0; i < vertex.numInEdges(); ++i) {
		unsigned other = vertex.getSourceVertexData(i);
		curr += other;
	}

	vertex.setData(curr);

	if (curr >= 10) {
		changed = false;
	}

	return changed;
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
	VType curr = vertex.data();
	outFile << vertex.globalId() << ": " << curr << std::endl;
    }

    ~WriterProgram() {
        outFile.close();
    }
};

int main(int argc, char* argv[]) {
    init();

    parse(&argc, argv, "--bm-tmpdir=", tmpDir);

    VType defaultVertex = 1;
    Engine<VType, EType>::init(argc, argv, defaultVertex);
    Engine<VType, EType>::signalAll();

    AggregateProgram<VType, EType> aggregateProgram;
    Engine<VType, EType>::run(&aggregateProgram, true);

    WriterProgram<VType, EType> writerProgram;
    Engine<VType, EType>::processAll(&writerProgram);

    Engine<VType, EType>::destroy();
    return 0;
}
