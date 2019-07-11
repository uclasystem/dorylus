#include <iostream>
#include "../engine/engine.hpp"
#include <fstream>
#include <vector>

using namespace std;

char tmpDir[256];

typedef float EType;
typedef vector<FeatType> VType;

template<typename VertexType, typename EdgeType>
class IncrementProgram : public VertexProgram<VertexType, EdgeType> {
public:
    bool update(Vertex<VertexType, EdgeType>& vertex, EngineContext& engineContext) {
	bool changed = false;
	VType curr = vertex.data();
	++curr[0];

	vertex.setData(curr);

	if (curr[0] <= 100) {
		changed = true;
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
	outFile << vertex.globalId() << ": ";
	for (FeatType f : vertex.data()) {
		outFile << f << " ";
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

    EType defaultEdge = 1888;
    VType defaultVertex = VType(2, 0);
    Engine<VType, EType>::init(argc, argv, defaultVertex, defaultEdge);
    Engine<VType, EType>::signalAll();
    
    IncrementProgram<VType, EType> incrementProgram;
    Engine<VType, EType>::run(&incrementProgram, true);

    WriterProgram<VType, EType> writerProgram;
    Engine<VType, EType>::processAll(&writerProgram);

    Engine<VType, EType>::destroy();
    return 0;
}
