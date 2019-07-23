#include <iostream>
#include <fstream>
#include <vector>
#include <cassert>
#include "../engine/engine.hpp"


using namespace std;


char tmpDir[256];


/** Define the aggregate vertex program. */
class AggregateProgram : public VertexProgram {

public:

    // Define my own update function that to be called in each iteration.
    void update(Vertex& vertex, unsigned layer) {
        VType curr = vertex.data();

        for (unsigned i = 0; i < vertex.getNumInEdges(); ++i) {
            vector<FeatType> other = vertex.getSourceVertexDataAt(i, layer);
            sumVectors(curr, other);
        }

        vertex.addData(curr);   // Push to the back instead of modify the value.
    }

private:

    // Sum up with a neighbors feature value vector.
    void sumVectors(vector<FeatType>& curr, vector<FeatType>& other) {
        assert(curr.size() <= other.size());
        for (int i = 0; i < curr.size(); ++i) {
            curr[i] += other[i];
        }
    }
};


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
        std::vector<VType>& data_all = vertex.dataAll();
        outFile << vertex.getGlobalId() << ": ";
        for (int i = 0; i < data_all.size(); ++i) {
            VType curr = data_all[i];
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
    VType defaultVertex = vector<FeatType>(2, 1);
    Engine::init(argc, argv, defaultVertex);

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
