#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <random>

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

void selectDeletions(std::string bSName, float probability) {
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

  std::string dbSName = bSName + ".dels";
  std::ofstream dbSStream;
  dbSStream.open(dbSName, std::ios::binary);

  dbSStream.write((char*) &header, sizeof(header));
  unsigned long long edgesSelected = 0;
  Dice dice;

  long long ignoreCount = 10000;

  VertexType srcdst[2];
  while(bSStream.read((char*) srcdst, header.sizeOfVertexType * 2)) {
    if(--ignoreCount > 0)
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
  if(argc < 2) {
    std::cout << "Dude! Invoke like this: " << argv[0] << " --bsfile=<filename>" << std::endl;
    return -1;
  }

  std::string bSFile;
  for(int i=0; i<argc; ++i) {
    if(strncmp("--bsfile=", argv[i], 9) == 0)
      bSFile = argv[i] + 9;
  }

  if(bSFile.size() == 0) {
    std::cout << "No BinarySnap file provided." << std::endl;
    return -1;
  }

  std::cout << "BinarySnap file: " << bSFile << std::endl;

  selectDeletions(bSFile, 0.05);

  return 0;
}
