#include <iostream>
#include "../engine/engine.hpp"
#include <fstream>
#include <memory>
#include <vector>

using namespace std;

char tmpDir[256];

typedef Empty EType;
typedef vector<FeatType> VType;

template<typename VertexType, typename EdgeType>
class PageRankProgram : public VertexProgram<VertexType, EdgeType> {
public:
    bool update(Vertex<VertexType, EdgeType>& vertex, EngineContext& engineContext) {
	bool changed = false;
	VType curr = vertex.data();

	vertex.setData(curr);

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
	outFile << vertex.globalId() << ": ";
    }

    ~WriterProgram() {
        outFile.close();
    }
};

int main(int argc, char* argv[]) {
    init();

    parse(&argc, argv, "--bm-tmpdir=", tmpDir);

    VType defaultVertex = vector<int>(2, 1);
    Engine<VType, EType>::init(argc, argv, defaultVertex);
    Engine<VType, EType>::signalAll();

//    InitProgram<VType, EType> initProg;
//    Engine<VType, EType>::quickRun(&initProg, false);

//    Engine<VType, EType>::signalAll();
    
    PageRankProgram<VType, EType> pagerankProgram;
    Engine<VType, EType>::run(&pagerankProgram, true);

    WriterProgram<VType, EType> writerProgram;
    Engine<VType, EType>::processAll(&writerProgram);

    Engine<VType, EType>::destroy();
    return 0;
}
