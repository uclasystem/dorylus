#include <iostream>
#include <fstream>
#include <vector>
#include <cassert>
#include <unistd.h>
#include "engine/engine.hpp"
#include "utils/utils.hpp"


/**
 *
 * Main entrance of the graph server logic.
 *
 */
int
main(int argc, char *argv[]) {

    // Initialize the engine.
    // The engine object is static and has been substantiated in Engine.cpp.
    engine.init(argc, argv);

    unsigned numEpochs = 30;
    unsigned valFreq = 1;

    if (engine.master())
        printLog(engine.getNodeId(),"Number of epochs: %u, validation frequency: %u",
                    numEpochs, valFreq);

    // Do specified number of epochs.
    for (unsigned epoch = 0; epoch < numEpochs; ++epoch) {
        printLog(engine.getNodeId(), "Starting Epoch %u", epoch+1);
        if (epoch != 0 && (epoch % valFreq == 0 || epoch == 1)) {
            if (engine.master())
                printLog(engine.getNodeId(), "Time for some validation");

            // Boolean of whether or not to run evaluation
            FeatType *predictData = engine.runForward();
            engine.runBackward(predictData);
        } else {
            FeatType *predictData = engine.runForward();
            // Do a backward-prop phase.
            if (engine.isGPUEnabled() == 0) {
                engine.runBackward(predictData);
            }
        }
    }

    // Procude the output files.
    engine.output();

    // Destroy the engine.
    engine.destroy();

    return 0;
}
