#include <iostream>
#include <fstream>
#include <vector>
#include <cassert>
#include "engine/engine.hpp"


using namespace std;


/**
 *
 * Main entrance of the aggregate benchmark.
 * 
 */
int
main(int argc, char *argv[]) {

    // Initialize the engine.
    std::vector<unsigned> layerConfig(5, 3);
    Engine::init(argc, argv, layerConfig);

    // Start one run of the engine.
    Engine::run();

    // Procude the output files.
    Engine::output();

    // Destroy the engine.
    Engine::destroy();

    return 0;
}
