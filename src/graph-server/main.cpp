#include <iostream>
#include <fstream>
#include <vector>
#include <cassert>
#include "engine/engine.hpp"


/**
 *
 * Main entrance of the aggregate benchmark.
 * 
 */
int
main(int argc, char *argv[]) {

    // Initialize the engine.
    Engine::init(argc, argv);

    // Start one run of the engine.
    Engine::run();

    // Procude the output files.
    Engine::output();

    // Destroy the engine.
    Engine::destroy();

    return 0;
}
