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


/**
 *
 * Read file header.
 * 
 */
void
readFile(std::string snapFile) {
	std::ifstream snapStream(snapFile);
    if (!snapStream.good())
        printf("Cannot open graph snap file: %s [Reason: %s]\n", snapFile.c_str(), std::strerror(errno));

    assert(snapStream.good());

	std::string line;
	while (std::getline(snapStream, line)) {
		if (line[0] == '#' || line[0] == '%')
			continue;

		std::istringstream iss(line);
		VertexType dst, src;

		if (!(iss >> src >> dst))
			break;

		if (dst == src)
			continue;

		header.numVertices = std::max(header.numVertices, std::max(src, dst));
		++header.numEdges;
	}

	snapStream.close();
	++header.numVertices;
}


/**
 *
 * Read the snap file and write binary snap file.
 * 
 */
void
readWriteFile(std::string snapFile, std::string bSFile, bool undirected, bool withheader) {
	std::ifstream snapStream(snapFile);
	if(!snapStream.good())
		printf("Cannot open graph snap file: %s [Reason: %s]\n", snapFile.c_str(), std::strerror(errno));
	
	assert(snapStream.good());

	std::ofstream bSStream;
	bSStream.open(bSFile, std::ios::binary);

	if (withheader)
		bSStream.write((char *) &header, sizeof(header));

	std::string line;
	unsigned long long oldId = 0;
	std::vector<unsigned long long> dsts;
	while (std::getline(snapStream, line)) {
		if (line[0] == '#' || line[0] == '%')
			continue;

		std::istringstream iss(line);
		VertexType src, dst;

		if (!(iss >> src >> dst))
			break;

		if(src == dst)
			continue;
	
		bSStream.write((char *) &src, header.sizeOfVertexType);
		bSStream.write((char *) &dst, header.sizeOfVertexType);

        if (undirected) {
            bSStream.write((char*) &dst, header.sizeOfVertexType);
            bSStream.write((char*) &src, header.sizeOfVertexType);
        }
	}
	
	snapStream.close();
	bSStream.close();
}


/**
 *
 * Main entrance.
 * 
 */
int
main(int argc, char* argv[]) {
	if(argc < 4) {
		std::cout << "Usage: " << argv[0] << " --snapfile=<BsnapFile> --undirected=<0/1> --header=<0/1>" << std::endl;
		return -1;
	}

	std::string snapFile;
	bool undirected = false;
	bool withheader = false;

	for (int i = 0; i < argc; ++i) {
		if (strncmp("--snapfile=", argv[i], 11) == 0) 
			snapFile = argv[i] + 11;
		if (strncmp("--undirected=", argv[i], 13) == 0) {
			int undir = 0;
			sscanf(argv[i] + 13, "%d", &undir);
			undirected = (undir == 0 ? false : true);
		}
		if (strncmp("--header=", argv[i], 9) == 0) {
			int hdr = 0;
			sscanf(argv[i] + 9, "%d", &hdr);
			withheader = (hdr == 0 ? false : true);
		}
	}

	if (snapFile.size() == 0) {
		std::cout << "Empty graph snap file." << std::endl;
		return -1; 
	} 

	std::cout << "SNAP file: " << snapFile << std::endl;
	std::cout << "Unidrected: " << (undirected ? "true" : "false") << std::endl;
	std::cout << "Header: " << (withheader ? "true" : "false") << std::endl;
	std::cout << "Self-edges will be removed..." << std::endl;
	std::cout << "If undirected, edge repitions might occur..." << std::endl;

	header.sizeOfVertexType = sizeof(VertexType);
	header.numVertices = 0;
	header.numEdges = 0;

	if (withheader) {
		readFile(snapFile);

        if (undirected) header.numEdges *= 2;
        std::cout << "Graph info - Vertices: " << header.numVertices
          << ", Edges: " << header.numEdges << std::endl;
    }

	std::string bSFile = snapFile + ".bsnap";
	readWriteFile(snapFile, bSFile, undirected, withheader);

	return 0;
}
