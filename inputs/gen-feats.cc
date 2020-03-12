#include <algorithm>
#include <cassert>
#include <condition_variable>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>
#include <set>


int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: gen-feats [numVerts] [numFeats] [datasetName]" << std::endl;
        return 13;
    }

    unsigned numVerts = std::atoi(argv[1]);
    unsigned numFeats = std::atoi(argv[2]);
    std::string fname = argv[3];

    std::default_random_engine dre;
    std::uniform_int_distribution<int> int_dist(numFeats/3, numFeats*3/4);
    std::uniform_int_distribution<int> int_dist2(0, numFeats-1);
    std::uniform_real_distribution<float> float_dist(-1.0, 1.0);

    std::ofstream featuresFile;
    featuresFile.open(fname, std::ofstream::out);

    for (uint32_t uv = 0; uv < numVerts; ++uv) {
        std::string line;
        std::set<unsigned> nz; // non zero
        unsigned numNz = int_dist(dre);
        while (nz.size() != numNz) {
            nz.insert(int_dist2(dre));
        }

        for (uint32_t uf = 0; uf < numFeats; ++uf) {
            if (nz.find(uf) != nz.end()) line += std::to_string(float_dist(dre));
            else line += "0";
            if (uf < numFeats-1) line += ", ";
        }
        line += "\n";

        featuresFile.write(line.c_str(), line.size());
    }
    featuresFile.close();

    return 0;
}
