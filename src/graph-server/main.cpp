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
    /*
    float splitPortion = 1.0 / 3.0;
    unsigned numEpochs = 2;
    unsigned valFreq = 10;

    if (Engine::master())
        printLog(Engine::getNodeId(), "% Train Data: %f, \
                    number of epochs: %u, validation frequency: %u",
                    splitPortion, numEpochs, valFreq);

    // Use one third of partitions as training and 2/3 as validation
    Engine::setTrainValidationSplit(1.0 / 3.0);
    printf("Here\n");
    // Do a forward-prop phase.
    for (unsigned epoch = 0; epoch < numEpochs; ++epoch) {
        printLog(Engine::getNodeId(), "Starting Epoch %u", epoch+1);
        if (epoch != 0 && (epoch % valFreq == 0 || epoch == 1)) {
            if (Engine::master())
                printLog(Engine::getNodeId(), "Time for some validation");

            // Boolean of whether or not to run evaluation
            Engine::runForward(false);

            Engine::makeBarrier();

            Engine::runBackward();
        } else {
            Engine::runForward();

            // Do a backward-prop phase.
            Engine::runBackward();
        }
    }
    */
    printf("Spliting\n");
    Engine::setTrainValidationSplit(1.0 / 3.0);
    printf("runForward\n");
    Engine::runForward(false);
    printf("runBorward\n");
    Engine::runBackward();

    Engine::runForward(true);

    // Do a backward-prop phase.
    Engine::runBackward();
    // Procude the output files.
    Engine::output();

    // Destroy the engine.
    Engine::destroy();

    return 0;
}
