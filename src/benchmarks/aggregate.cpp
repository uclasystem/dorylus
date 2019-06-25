#include <iostream>
#include "../engine/engine.hpp"
#include <fstream>
#include <vector>

using namespace std;

char tmpDir[256];

typedef Empty EType;

typedef struct vType {
	std::vector<int> features;
	int iter;

	vType() { features = std::vector<int>(2, 0); iter = 0; }
	vType(int n) { features = std::vector<int>(2, n); iter = 0; }
} VType;

//template<typename VertexType, typename EdgeType>
//class InitProgram : public VertexProgram<VertexType, EdgeType> {
//public:
//	bool update(Vertex<VertexType, EdgeType>& vertex, EngineContext& engineContext) {
//		vertex.setData( VType(vertex.globalIdx) );
//
//		return false;
//	}
//};

template<typename VertexType, typename EdgeType>
class PageRankProgram : public VertexProgram<VertexType, EdgeType> {
public:
    bool update(Vertex<VertexType, EdgeType>& vertex, EngineContext& engineContext) {
	bool changed = false;
	VType curr = vertex.data();

//	for (int& n : curr.features) {
//		++n;
//	}
	++curr.iter;
	std::cout << "Current Iter: " << curr.iter << std::endl;

	vertex.setData(curr);

//	if (curr.iter <= 10) {
//		changed = true;
//	}

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
	//for (int n : curr.features) {
	//	outFile << n << " ";
	//}
	//outFile << std::endl;
    }

    ~WriterProgram() {
        outFile.close();
    }
};

int main(int argc, char* argv[]) {
    init();

    parse(&argc, argv, "--bm-tmpdir=", tmpDir);

    VType defaultVertex;
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
