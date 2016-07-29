#include <iostream>
#include "../engine/engine.hpp"
#include <fstream>

typedef Empty EType;
typedef char VType;

char tmpDir[255];

template<typename VertexType, typename EdgeType>
class WriterProgram : public VertexProgram<VertexType, EdgeType> {
    std::ofstream outFile;
public:
    WriterProgram() {
        char filename[300];
        sprintf(filename, "%s/output_%u", tmpDir, NodeManager::getNodeId()); 
        outFile.open(filename);
        fprintf(stderr, "Writing degrees to: %s\n", filename);
    }

    void processVertex(Vertex<VertexType, EdgeType>& vertex) {
       outFile << vertex.globalId() << " " << vertex.numInEdges() << std::endl; 
    }

    ~WriterProgram() {
        outFile.close();
    }
};

int main(int argc, char* argv[]) {
    init();

    assert(parse(&argc, argv, "--bm-tmpdir=", tmpDir));
    fprintf(stderr, "tmpDir = %s\n", tmpDir);

    VType defaultVertex = '.';
    Engine<VType, EType>::init(argc, argv, defaultVertex);
    
    WriterProgram<VType, EType> writerProgram;
    Engine<VType, EType>::processAll(&writerProgram);

    Engine<VType, EType>::destroy();
    return 0;
}
