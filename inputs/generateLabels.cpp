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
        std::cerr << "Usage: gen-feats [numVerts] [numLabels] [datasetName]" << std::endl;
        return 13;
    }

    unsigned numVerts = std::atoi(argv[1]);
    unsigned numLabels = std::atoi(argv[2]);
    std::string fname = argv[3];

    std::default_random_engine dre;
    std::uniform_int_distribution<int> int_dist(0, numLabels);

    std::ofstream labelsFile;
    labelsFile.open(fname + ".labels", std::ios::binary);

    labelsFile.write((char*) &numLabels, sizeof(unsigned));
    for (uint32_t uv = 0; uv < numVerts; ++uv) {
        unsigned label = int_dist(dre);
        labelsFile.write((char*) &label, sizeof(unsigned));
    }
    labelsFile.close();

    return 0;
}
