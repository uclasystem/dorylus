#include "utils.hpp"
#include <cmath>

GPUTimers gtimers;
FILE* outputFile;
std::mutex fileMutex;

Chunk createChunk(unsigned rows, unsigned nChunks, unsigned id, unsigned globalId, unsigned layer,
  PROP_TYPE dir, unsigned ep, bool vertex) {
    unsigned partRows = std::ceil((float)rows / (float)nChunks);
    unsigned lowBound = id * partRows;
    unsigned upBound = (id + 1) * partRows;
    if (upBound > rows) upBound = rows;

    return Chunk{id, globalId, lowBound, upBound, layer, dir, ep, vertex};
}

std::ofstream matrixFile;
std::mutex mFileMutex;

void matrixToFile(std::string name, FeatType* fptr, unsigned start, unsigned end, unsigned c) {
    matrixFile.open(name, std::ofstream::trunc | std::ofstream::out);
    std::string out = "";
    for (uint32_t u = start; u < end; ++u) {
        for (uint32_t uj = 0; uj < c; ++uj) {
            out += std::to_string(fptr[u * c + uj]);
            if (uj < c-1) out += " ";
        }
        out += "\n";
    }
    mFileMutex.lock();
    matrixFile.write(out.c_str(), out.size());
    mFileMutex.unlock();
    matrixFile.close();
}

unsigned getAbsLayer(const Chunk &chunk, unsigned numLayers) {
    // YIFAN: I set the "numLayers" to 10 here to avoid any conflicts
    return chunk.dir == PROP_TYPE::FORWARD ? (chunk.layer) : (2 * numLayers - 1 - chunk.layer);
}

Chunk incLayer(const Chunk &chunk, unsigned numLayers) {
    Chunk nextChunk = chunk;
    if (chunk.dir == PROP_TYPE::FORWARD && chunk.layer < numLayers - 1) {
        // Keep dir as FORWARD and inc layer
        nextChunk.layer++;
    } else if (chunk.dir == PROP_TYPE::FORWARD && chunk.layer == numLayers - 1) {
        // Change dir to BACKWARD and dec layer (the final layer backawrd lambda is merged into the forward lambda)
        nextChunk.dir = PROP_TYPE::BACKWARD;
        nextChunk.layer--;
    } else if (chunk.dir == PROP_TYPE::BACKWARD && chunk.layer > 0) {
        // Keep dir as BACKWARD and dec layer
        nextChunk.layer--;
    } else if (chunk.dir == PROP_TYPE::BACKWARD && chunk.layer == 0) {
        // Change dir to FORWARD and inc epoch
        nextChunk.dir = PROP_TYPE::FORWARD;
        nextChunk.epoch++;
    }
    return nextChunk;
}

bool isLastLayer(const Chunk &chunk) {
    return chunk.dir == PROP_TYPE::BACKWARD && chunk.layer == 0;
}
