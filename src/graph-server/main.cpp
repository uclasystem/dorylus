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

    // Do a forward-prop phase.
    Engine::runForward();

    // Do a backward-prop phase.
    // Engine::runBackward();

    // Procude the output files.
    Engine::output();

    // Destroy the engine.
    Engine::destroy();

    return 0;
}
