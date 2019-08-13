#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <vector>
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/trim.hpp>


using namespace std;


typedef float FeatType;


struct FeaturesHeader{
    unsigned int numFeautures;
};
static FeaturesHeader head;


// TODO: verify written file
// TODO: add header


/**
 *
 * Read in features file, convert into binary representation, and write to a '.bsnap' file.
 * 
 */
void
readWriteFile(std::string featuresFileName) {
    std::ifstream infile(featuresFileName.c_str());
    if (!infile.good())
        printf("Cannot open feature file: %s [Reason: %s]\n", featuresFileName.c_str(), std::strerror(errno));

    assert(infile.good());

    std::ofstream bSStream;
    bSStream.open(featuresFileName + ".bsnap", std::ios::binary);
    bSStream.write(reinterpret_cast<char *>(&head), sizeof(FeaturesHeader));

    std::string line;
    while (!infile.eof()) {
        std::getline(infile, line);
        boost::algorithm::trim(line);

        if (line[0] < '0' || line[0] > '9')
            continue;

        std::vector<std::string> splited_strings;
        std::vector<FeatType> feature_vec;

        // Split each line into numbers.
        boost::split(splited_strings, line, boost::is_any_of(", "), boost::token_compress_on);
        assert(size_t(head.numFeautures) == splited_strings.size());

        for (std::string& substr : splited_strings) {
            FeatType f = std::stof(substr);
            bSStream.write(reinterpret_cast<char *>(&f), sizeof(FeatType));
        }
    }
}


/**
 *
 * Main entrance.
 * 
 */
int
main(int argc, char *argv[]) {
    if (argc != 3) {
        std::cout << "Usage: " << argv[0] << " --featuresfile=<FeatureFile> --featuredimension=<FeatureDimension>" << std::endl;
        return -1;
    }

    std::string featuresFile;
    bool withheader = false;
    for (int i = 0; i < argc; ++i) {
        if (strncmp("--featuresfile=", argv[i], 15) == 0)
            featuresFile = argv[i] + 15;
        if (strncmp("--featuredimension=", argv[i], 19) == 0)
            sscanf(argv[i] + 19, "%u", &head.numFeautures);
    }
    std::cout << "Features file: " << featuresFile << std::endl;
    std::cout << "Features size: " << head.numFeautures << std::endl;

    assert(head.numFeautures > 0);

    if (featuresFile.size() == 0) {
        std::cerr << "Empty features file." << std::endl;
        return -1;
    }

    readWriteFile(featuresFile);

    return 0;
}
