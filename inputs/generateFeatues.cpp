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


/**
 * Writes a features file directly to binary format ready to be used by GNN-Lambda
 */
int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: gen-feats <numVerts> <numFeats> <datasetName> [dense]" << std::endl;
        return 13;
    }

    unsigned numVerts = std::atoi(argv[1]);
    unsigned numFeats = std::atoi(argv[2]);
    std::string fname = argv[3];
    bool dense = false;
    if (argc >= 5) dense = std::string(argv[4]) == "1";

    std::default_random_engine dre;
    std::uniform_int_distribution<int> int_dist(numFeats/3, numFeats*3/4);
    std::uniform_int_distribution<int> int_dist2(0, numFeats-1);
    std::uniform_real_distribution<float> float_dist(-1.0, 1.0);

    std::ofstream featuresFile;
    featuresFile.open(fname + ".feats", std::ios::binary);
    featuresFile.write((char*) &numFeats, sizeof(unsigned));

    for (uint32_t uv = 0; uv < numVerts; ++uv) {
        std::vector<float> features;
        std::set<unsigned> nz; // non zero
        unsigned numNz = int_dist(dre);
        while (nz.size() != numNz) {
            nz.insert(int_dist2(dre));
        }

        for (uint32_t uf = 0; uf < numFeats; ++uf) {
            if (nz.find(uf) != nz.end()) features.push_back(float_dist(dre));
            else features.push_back(0.0);
        }

        featuresFile.write((char*) features.data(), features.size() * sizeof(float));
    }
    featuresFile.close();

    return 0;
}
