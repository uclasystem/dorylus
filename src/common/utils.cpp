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
