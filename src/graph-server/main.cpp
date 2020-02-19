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
    engine.setPipeline(false);

    if (engine.master())
        printLog(engine.getNodeId(),"Number of epochs: %u, validation frequency: %u",
                    numEpochs, valFreq);

    // Do specified number of epochs.
    Timer epochTimer;
    for (unsigned epoch = 0; epoch < numEpochs; ++epoch) {
        epochTimer.start();
        printLog(engine.getNodeId(), "Starting Epoch %u", epoch+1);
        if (epoch != 0 && (epoch % valFreq == 0 || epoch == 1)) {
            FeatType *predictData = engine.runForward(epoch);
            engine.runBackward(predictData);
        } else {
            FeatType *predictData = engine.runForward(epoch);
            // Do a backward-prop phase.
            engine.runBackward(predictData);
        }
        epochTimer.stop();

        printLog(engine.getNodeId(), "Time for epoch %u: %f ms",
                 epoch+1, epochTimer.getTime());
        engine.addEpochTime(epochTimer.getTime());
    }

    // Procude the output files.
    printLog(engine.getNodeId(), "Running output");
    engine.output();

    // Destroy the engine.
    printLog(engine.getNodeId(), "Running destroy");
    engine.destroy();

    printLog(engine.getNodeId(), "Returning");
    return 0;
}
