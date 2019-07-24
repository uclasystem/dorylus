#include <iostream>
#include <fstream>
#include <vector>
#include <cassert>
#include "../engine/engine.hpp"


using namespace std;


char tmpDir[256];


/** Define the writer vertex program for output. */
class WriterProgram : public VertexProgram {

private:

    std::ofstream outFile;

public:

    WriterProgram() {
        char filename[200];
        sprintf(filename, "%s/output_%u", tmpDir, NodeManager::getNodeId()); 
        outFile.open(filename);
    }

    ~WriterProgram() {
        outFile.close();
    }

    // Define the output generated for each vertex.
    void processVertex(Vertex& vertex) {
        std::vector<VertexType>& data_all = vertex.dataAll();
        outFile << vertex.getGlobalId() << ": ";
        for (int i = 0; i < data_all.size(); ++i) {
            VertexType curr = data_all[i];
            for (int j = 0; j < curr.size(); ++j)
                outFile << curr[j] << " ";
            outFile << "| ";
        }
        outFile << std::endl;
    }
};


/**
 *
 * Main entrance of the aggregate benchmark.
 * 
 */
int
main(int argc, char *argv[]) {
    initArgs();

    parseArgs(&argc, argv, "--bm-tmpdir=", tmpDir);

    // Initialize the engine.
    std::vector<unsigned> layerConfig(5, 3);
    Engine::init(argc, argv, layerConfig);

    // Start one run of the engine, on the aggregate program.
    AggregateProgram aggregateProgram;
    Engine::run(&aggregateProgram, true);

    // Procude the output files using the writer program.
    WriterProgram writerProgram;
    Engine::processAll(&writerProgram);

    // Destroy the engine.
    Engine::destroy();

    return 0;
}
