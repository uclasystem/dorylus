#include <iostream>
#include <fstream>
#include <vector>
#include <cassert>
#include <unistd.h>
#include "engine/engine.hpp"
#include "utils/utils.hpp"


/**
 *
 * Main entrance of the aggregate benchmark.
 * 
 */
int
main(int argc, char *argv[]) {

    // Initialize the engine.
    Engine::init(argc, argv);

    // Use one third of partitions as training and 2/3 as validation
    Engine::setTrainValSplit(1.0 / 3.0);

    // Do a forward-prop phase.
    for (unsigned epoch = 0; epoch < Engine::getNumEpochs(); ++epoch) {
        printLog(Engine::getNodeId(), "Starting Epoch %u", epoch+1);
        if (epoch != 0 && epoch % Engine::getValFreq() == 0) {
            if (Engine::master())
                printLog(Engine::getNodeId(), "Time for some validation");

            // Boolean of whether or not to run evaluation
            Engine::runForward(true);
        } else {
            Engine::runForward();

            // Do a backward-prop phase.
            if(Engine::isGPUEnabled()==0)
                Engine::runBackward();
        }
    }

    // Procude the output files.
    Engine::output();

    // Destroy the engine.
    Engine::destroy();

    return 0;
}
