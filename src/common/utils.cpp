#include "utils.hpp"

GPUTimers gtimers;

std::ofstream debugFile;
std::mutex fileMutex;

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
