#include <iostream>
#include "../engine/engine.hpp"
#include <fstream>
#include <vector>

using namespace std;

char tmpDir[256];

typedef Empty EType;
typedef int VType;

template<typename VertexType, typename EdgeType>
class PageRankProgram : public VertexProgram<VertexType, EdgeType> {
public:
    bool update(Vertex<VertexType, EdgeType>& vertex, EngineContext& engineContext) {
	bool changed = false;
	int curr = vertex.data();
	++curr;

	if (curr >= 10) {
		changed = true;
	}

	return changed;
    }

private:
    void sumVectors(vector<int>& v, const vector<int>& v1) {
	for (int i = 0; i < v1.size(); ++i) {
		v[i] += v1[i];
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
	outFile << vertex.globalId() << ": " << vertex.data() << std::endl;
    }

    ~WriterProgram() {
        outFile.close();
    }
};

int main(int argc, char* argv[]) {
    init();

    parse(&argc, argv, "--bm-tmpdir=", tmpDir);

    VType defaultVertex = 1.0;
    Engine<VType, EType>::init(argc, argv, defaultVertex);
    Engine<VType, EType>::signalAll();
    
    PageRankProgram<VType, EType> pagerankProgram;
    Engine<VType, EType>::run(&pagerankProgram, true);

    WriterProgram<VType, EType> writerProgram;
    Engine<VType, EType>::processAll(&writerProgram);

    Engine<VType, EType>::destroy();
    return 0;
}
