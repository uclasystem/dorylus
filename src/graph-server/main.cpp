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
    unsigned numEpochs = engine.getNumEpochs();
    unsigned valFreq = 1;

    if (engine.master())
        printLog(engine.getNodeId(),"Number of epochs: %u, validation frequency: %u",
                    numEpochs, valFreq);
    // Sync all nodes before starting computation
    engine.makeBarrier();

    // Do specified number of epochs.
    Timer epochTimer;
    engine.runGCN();
//j    for (unsigned epoch = 0; epoch < numEpochs; ++epoch) {
//j        epochTimer.start();
//j        printLog(engine.getNodeId(), "Starting Epoch %u", epoch+1);
//j        if (epoch != 0 && (epoch % valFreq == 0 || epoch == 1)) {
//j            FeatType *predictData = engine.runForward(epoch);
//j            engine.runBackward(predictData);
//j        } else {
//j            FeatType *predictData = engine.runForward(epoch);
//j            // Do a backward-prop phase.
//j            engine.runBackward(predictData);
//j        }
//j        epochTimer.stop();
//j
//j        printLog(engine.getNodeId(), "Time for epoch %u: %f ms",
//j                 epoch+1, epochTimer.getTime());
//j        engine.addEpochTime(epochTimer.getTime());
//j    }

    // Procude the output files.
    engine.output();

    usleep(100000);

    // Destroy the engine.
    engine.destroy();

    return 0;
}
