#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

typedef unsigned VertexType;

struct HeaderType {
  int sizeOfVertexType;
  VertexType numVertices;
  unsigned long long numEdges;
};

HeaderType header;

class Dice {
  std::random_device rd;
  std::mt19937 gen;
  std::uniform_int_distribution<> dis;
  int lower;
  int upper;

  public:
  Dice(int lb = 1, int ub = 100) : gen(rd()), dis(lb, ub), lower(lb), upper(ub) { assert(ub > lb); }

  bool chance(float probability) {
    int expected = lower + int(probability * (upper - lower + 1)) - 1;
    return (dis(gen) <= expected);
  }
};

void selectDeletions(std::string bSName, std::string degName, float probability, int theshold) {
  std::ifstream bSStream;
  bSStream.open(bSName, std::ios::binary);

  if(!bSStream.good()) {
    std::cout << "Cannot open " << bSName << std::endl;
    abort();
  }

  bSStream.read((char*) &header, sizeof(header));

  assert(header.sizeOfVertexType == sizeof(VertexType));

  std::cout << "# Vertices: " << header.numVertices << " Edges: " << header.numEdges << std::endl;
  header.numEdges *= probability;
  std::cout << "# Vertices: " << header.numVertices << " Deletion Edges: " << header.numEdges << std::endl;

  std::vector<unsigned> degrees(header.numVertices, 0);
  std::ifstream degStream(degName);
  if(!degStream.good()) {
    std::cout << "Cannot open " << degName << std::endl;
    abort();
  }

  VertexType v; unsigned d;
  while(degStream >> v >> d) {
    degrees[v] = d;
  }

  degStream.close();

  std::string dbSName = bSName + ".dels";
  std::ofstream dbSStream;
  dbSStream.open(dbSName, std::ios::binary);

  dbSStream.write((char*) &header, sizeof(header));
  unsigned long long edgesSelected = 0;
  Dice dice;

  long long ignoreCount = 0;

  VertexType srcdst[2];
  while(bSStream.read((char*) srcdst, header.sizeOfVertexType * 2)) {
    if(--ignoreCount > 0)
      continue;

    if(degrees[srcdst[0]] > theshold)
      continue;

    if(dice.chance(probability)) {
      dbSStream.write((char*) srcdst, header.sizeOfVertexType * 2);

      ++edgesSelected;
      if(edgesSelected == header.numEdges)
        break;
    }
  }

  srcdst[0] = 0; srcdst[1] = 0;
  while(edgesSelected != header.numEdges) {
    dbSStream.write((char*) srcdst, header.sizeOfVertexType * 2);
    ++edgesSelected;
  }

  bSStream.close();
  dbSStream.close();
}

int main(int argc, char* argv[]) {
  if(argc < 5) {
    std::cout << "Dude! Invoke like this: " << argv[0] << " --bsfile=<filename> --degreefile=<degreename> --probability=<number> --threshold=<number>" << std::endl;
    return -1;
  }

  std::string bSFile, degFile;
  float probability = 0.1;
  int threshold = 100;
  for(int i=0; i<argc; ++i) {
    if(strncmp("--bsfile=", argv[i], 9) == 0)
      bSFile = argv[i] + 9;

    if(strncmp("--degreefile=", argv[i], 13) == 0)
      degFile = argv[i] + 13;

    if(strncmp("--probability=", argv[i], 14) == 0)
      probability = atof(argv[i] + 14);

    if(strncmp("--threshold=", argv[i], 12) == 0)
      threshold = atoi(argv[i] + 12);
  }

  if(bSFile.size() == 0 || degFile.size() == 0) {
    std::cout << "No BinarySnap file provided." << std::endl;
    return -1;
  }

  std::cout << "BinarySnap file: " << bSFile << std::endl;
  std::cout << "Degree file: " << degFile << std::endl;
  std::cout << "Probability: " << probability << std::endl;
  std::cout << "Thresold: " << threshold << std::endl;

  selectDeletions(bSFile, degFile, probability, threshold);

  return 0;
}
