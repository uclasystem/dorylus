#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <vector>

typedef unsigned VertexType;
typedef unsigned CountType;

struct HeaderType {
	int sizeOfVertexType;
	VertexType numVertices;
	unsigned long long numEdges;
};

HeaderType header;


void readFile(std::string snapFile) {
	std::ifstream snapStream(snapFile);

	if(!snapStream.good()) {
		std::cout << "Cannot open " << snapFile << std::endl;
		abort();
	}

	std::string line;
	while (std::getline(snapStream, line)) {
		if(line[0] == '#' || line[0] == '%')
			continue;

		std::istringstream iss(line);
		VertexType dst, src;
		if (!(iss >> src >> dst)) { break; }

		if(dst == src)
			continue;

		header.numVertices = std::max(header.numVertices, std::max(src, dst)); 
		++header.numEdges;
	}
	snapStream.close();
	++header.numVertices;
}

void readWriteFile(std::string snapFile, std::string bSFile, bool undirected, bool withheader) {
	std::ifstream snapStream(snapFile);
	if(!snapStream.good()) {
		std::cout << "Cannot open " << snapFile << std::endl;
		abort();
	}

	std::ofstream bSStream;
	bSStream.open(bSFile, std::ios::binary);

	if(undirected)
		header.numEdges *= 2;

	if(withheader) {
		bSStream.write((char*) &header, sizeof(header));
	}

	std::string line;
	unsigned long long oldId = 0;
	std::vector<unsigned long long> dsts;
	while (std::getline(snapStream, line)) {
		if(line[0] == '#' || line[0] == '%')
			continue;

		std::istringstream iss(line);
		VertexType src, dst;
		if (!(iss >> src >> dst)) { break; }

		if(src == dst)
			continue;
	
		bSStream.write((char*) &src, header.sizeOfVertexType);
		bSStream.write((char*) &dst, header.sizeOfVertexType);
	}
	
	snapStream.close();
	bSStream.close();
}

int main(int argc, char* argv[]) {

	if(argc < 4) {
		std::cout << "Usage: " << argv[0] << " --snapfile=<snapfile> --undirected=<0|1> --header=<0|1>" << std::endl;
		return -1;
	}

	std::string snapFile;
	bool undirected = false;
	bool withheader = false;

	for(int i=0; i<argc; ++i) {
		if(strncmp("--snapfile=", argv[i], 11) == 0) 
			snapFile = argv[i] + 11;
		if(strncmp("--undirected=", argv[i], 13) == 0) {
			int undir = 0;
			sscanf(argv[i] + 13, "%d", &undir);
			undirected = (undir == 0 ? false : true);
		}
		if(strncmp("--header=", argv[i], 9) == 0) {
			int hdr = 0;
			sscanf(argv[i] + 9, "%d", &hdr);
			withheader = (hdr == 0 ? false : true);
		}
	}

	if(snapFile.size() == 0) {
		std::cout << "No SNAP file provided (--snapfile)." << std::endl;
		return -1; 
	} 

	std::cout << "SNAP file: " << snapFile << std::endl;
	std::cout << "Unidrected: " << (undirected ? "true" : "false") << std::endl;
	std::cout << "Header: " << (withheader ? "true" : "false") << std::endl;
	std::cout << "Self-edges will be removed" << std::endl;
	std::cout << "If undirected, edge repitions might occur" << std::endl;

	header.sizeOfVertexType = sizeof(VertexType);
	header.numVertices = 0;
	header.numEdges = 0;

	if(withheader)
		readFile(snapFile);

	std::string bSFile = snapFile + ".bsnap";
	readWriteFile(snapFile, bSFile, undirected, withheader);

	return 0;
}
