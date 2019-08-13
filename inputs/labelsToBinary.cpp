#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <vector>
#include <typeinfo>
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/trim.hpp>


using namespace std;


typedef unsigned LabelType;


struct LabelsHeader {
    LabelType labelKinds;
};
static LabelsHeader head;


// TODO: verify written file
// TODO: add header


/**
 *
 * Read in labels file, convert into binary representation, and write to a '.bsnap' file.
 * 
 */
void
readWriteFile(std::string labelsFileName) {
    std::ifstream infile(labelsFileName.c_str());
    if (!infile.good())
        printf("Cannot open labels file: %s [Reason: %s]\n", labelsFileName.c_str(), std::strerror(errno));

    assert(infile.good());

    std::ofstream bSStream;
    bSStream.open(labelsFileName + ".bsnap", std::ios::binary);
    bSStream.write(reinterpret_cast<char *>(&head), sizeof(LabelsHeader));

    std::string line;
    while (!infile.eof()) {
        std::getline(infile, line);
        boost::algorithm::trim(line);

        if (line.length() > 0) {
            if (line[0] < '0' || line[0] > '9')
                continue;

            LabelType label = std::stoul(line);
            bSStream.write(reinterpret_cast<char *>(&label), sizeof(LabelType));
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
        std::cout << "Usage: " << argv[0] << " --labelsfile=<LabelFile> --labelkinds=<LabelKinds>" << std::endl;
        return -1;
    }

    std::string labelsFile;
    bool withheader = false;
    for (int i = 0; i < argc; ++i) {
        if (strncmp("--labelsfile=", argv[i], 13) == 0)
            labelsFile = argv[i] + 13;
        if (strncmp("--labelkinds=", argv[i], 13) == 0)
            sscanf(argv[i] + 13, "%u", &head.labelKinds);
    }
    std::cout << "Labels file: " << labelsFile << std::endl;
    std::cout << "Label kinds: " << head.labelKinds << std::endl;

    assert(head.labelKinds > 0);

    if (labelsFile.size() == 0) {
        std::cerr << "Empty labels file." << std::endl;
        return -1;
    }

    readWriteFile(labelsFile);

    return 0;
}
