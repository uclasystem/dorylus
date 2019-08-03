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

        if(line[0] < '0' || line[0] > '9')
            continue;

        std::vector<std::string> splited_strings;
        std::vector<float> feature_vec;

        // Split each line into numbers.
        boost::split(splited_strings, line, boost::is_any_of(", "), boost::token_compress_on);
        assert(size_t(head.numFeautures) == splited_strings.size());

        for (std::string& substr : splited_strings) {
            float f = std::stof(substr);
            bSStream.write(reinterpret_cast<char *>(&f), sizeof(float));
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
        std::cout << "Usage: " << argv[0] << " --featurefile=<FeatureFile> --featuredimension=<FeatureDimension>" << std::endl;
        return -1;
    }

    std::string featureFile;
    bool withheader = false;
    for (int i = 0; i < argc; ++i) {
        if (strncmp("--featurefile=", argv[i], 14) == 0)
            featureFile = argv[i] + 14;
        if (strncmp("--featuredimension=", argv[i], 19) == 0)
            sscanf(argv[i] + 16, "%u", &head.numFeautures);
    }
    std::cout << "Feature file: " << featureFile << std::endl;
    std::cout << "Feature size: " << head.numFeautures << std::endl;

    assert(head.numFeautures > 0);

    if (featureFile.size() == 0) {
        std::cerr << "Empty feature file." << std::endl;
        return -1;
    }

    readWriteFile(featureFile);

    return 0;
}
