#ifndef __ENGINE_HPP__
#define __ENGINE_HPP__

#include "engine.h"
#include "../commmanager/commmanager.h"
#include "../nodemanager/nodemanager.h"
#include "ghostvertex.hpp"
#include <fstream>
#include <cmath>
#include <boost/program_options.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/algorithm/string/classification.hpp> // Include boost::for is_any_of.
#include <boost/algorithm/string/split.hpp>
#include <string>
#include <stdlib.h>
#include <omp.h>


template <typename VertexType, typename EdgeType>
Graph<VertexType, EdgeType> Engine<VertexType, EdgeType>::graph;

template <typename VertexType, typename EdgeType>
ThreadPool* Engine<VertexType, EdgeType>::dataPool = NULL;

template <typename VertexType, typename EdgeType>
unsigned Engine<VertexType, EdgeType>::dThreads = NUM_DATA_THREADS;

template <typename VertexType, typename EdgeType>
ThreadPool* Engine<VertexType, EdgeType>::computePool = NULL;

template <typename VertexType, typename EdgeType>
unsigned Engine<VertexType, EdgeType>::cThreads = NUM_COMP_THREADS;

template <typename VertexType, typename EdgeType>
std::string Engine<VertexType, EdgeType>::graphFile;

template <typename VertexType, typename EdgeType>
std::string Engine<VertexType, EdgeType>::featuresFile;

template <typename VertexType, typename EdgeType>
unsigned Engine<VertexType, EdgeType>::poFrequency = PUSHOUT_FREQUENCY;

template <typename VertexType, typename EdgeType>
BitsetScheduler* Engine<VertexType, EdgeType>::scheduler = NULL;

template <typename VertexType, typename EdgeType>
DenseBitset* Engine<VertexType, EdgeType>::shadowScheduler = NULL;

template <typename VertexType, typename EdgeType>
DenseBitset* Engine<VertexType, EdgeType>::trimScheduler = NULL;

template <typename VertexType, typename EdgeType>
VertexProgram<VertexType, EdgeType>* Engine<VertexType, EdgeType>::vertexProgram = NULL;

template <typename VertexType, typename EdgeType>
EdgeType (*Engine<VertexType, EdgeType>::edgeWeight) (IdType, IdType) = NULL;

template <typename VertexType, typename EdgeType>
IdType Engine<VertexType, EdgeType>::currId = 0;

template <typename VertexType, typename EdgeType>
Lock Engine<VertexType, EdgeType>::lockCurrId;

template <typename VertexType, typename EdgeType>
unsigned Engine<VertexType, EdgeType>::nodeId;

template <typename VertexType, typename EdgeType>
unsigned Engine<VertexType, EdgeType>::numNodes;

template <typename VertexType, typename EdgeType>
bool Engine<VertexType, EdgeType>::die = false;

template <typename VertexType, typename EdgeType>
bool Engine<VertexType, EdgeType>::compDone = false;

template <typename VertexType, typename EdgeType>
bool Engine<VertexType, EdgeType>::halt = false;

template <typename VertexType, typename EdgeType>
pthread_mutex_t Engine<VertexType, EdgeType>::mtxCompWaiter;

template <typename VertexType, typename EdgeType>
pthread_cond_t Engine<VertexType, EdgeType>::condCompWaiter;

template <typename VertexType, typename EdgeType>
pthread_mutex_t Engine<VertexType, EdgeType>::mtxDataWaiter;

template <typename VertexType, typename EdgeType>
pthread_cond_t Engine<VertexType, EdgeType>::condDataWaiter;

template <typename VertexType, typename EdgeType>
Barrier Engine<VertexType, EdgeType>::barComp;

template <typename VertexType, typename EdgeType>
Barrier Engine<VertexType, EdgeType>::barCompData;

template <typename VertexType, typename EdgeType>
pthread_barrier_t Engine<VertexType, EdgeType>::barDebug;

template <typename VertexType, typename EdgeType>
std::atomic<unsigned> Engine<VertexType, EdgeType>::remPushOut = ATOMIC_VAR_INIT(0);

template <typename VertexType, typename EdgeType>
EngineContext Engine<VertexType, EdgeType>::engineContext;

template <typename VertexType, typename EdgeType>
VertexType Engine<VertexType, EdgeType>::defaultVertex;

template <typename VertexType, typename EdgeType>
EdgeType Engine<VertexType, EdgeType>::defaultEdge;

template <typename VertexType, typename EdgeType>
unsigned Engine<VertexType, EdgeType>::iteration = 0;

template <typename VertexType, typename EdgeType>
unsigned Engine<VertexType, EdgeType>::curr_layer = 1;

template <typename VertexType, typename EdgeType>
bool Engine<VertexType, EdgeType>::undirected = false;

template <typename VertexType, typename EdgeType>
bool Engine<VertexType, EdgeType>::firstIteration = true;

template <typename VertexType, typename EdgeType>
double Engine<VertexType, EdgeType>::timProcess = 0.0;

template <typename VertexType, typename EdgeType>
double Engine<VertexType, EdgeType>::allTimProcess = 0.0;

template <typename VertexType, typename EdgeType>
double Engine<VertexType, EdgeType>::timInit = 0.0;

template <typename VertexType, typename EdgeType>
std::vector<std::tuple<unsigned long long, IdType, IdType> > Engine<VertexType, EdgeType>::insertStream;

template <typename VertexType, typename EdgeType>
unsigned long long Engine<VertexType, EdgeType>::globalInsertStreamSize = 0;

template <typename VertexType, typename EdgeType>
std::vector<std::tuple<unsigned long long, IdType, IdType> > Engine<VertexType, EdgeType>::deleteStream;

template <typename VertexType, typename EdgeType>
unsigned long long Engine<VertexType, EdgeType>::globalDeleteStreamSize = 0;

template <typename VertexType, typename EdgeType>
unsigned Engine<VertexType, EdgeType>::baseEdges = 0;

template <typename VertexType, typename EdgeType>
unsigned Engine<VertexType, EdgeType>::numBatches = 0;

template <typename VertexType, typename EdgeType>
unsigned Engine<VertexType, EdgeType>::batchSize = 0;

template <typename VertexType, typename EdgeType>
unsigned Engine<VertexType, EdgeType>::deletePercent = 0;

template <typename VertexType, typename EdgeType>
InOutType Engine<VertexType, EdgeType>::onAdd = NEITHER;

template <typename VertexType, typename EdgeType>
InOutType Engine<VertexType, EdgeType>::onDelete = NEITHER;

template <typename VertexType, typename EdgeType>
void (*Engine<VertexType, EdgeType>::onAddHandler) (VertexType& v) = NULL;

template <typename VertexType, typename EdgeType>
void (*Engine<VertexType, EdgeType>::onDeleteHandler) (VertexType& v) = NULL;

template <typename VertexType, typename EdgeType>
void (*Engine<VertexType, EdgeType>::onDeleteSmartHandler) (VertexType& v, LightEdge<VertexType, EdgeType>& e) = NULL;

template <typename VertexType, typename EdgeType>
void Engine<VertexType, EdgeType>::parseArgs(int argc, char* argv[]) {
  boost::program_options::options_description desc("Allowed options");
  desc.add_options()
    ("help", "Produce help message")
    ("config", boost::program_options::value<std::string>()->default_value(std::string(DEFAULT_CONFIG_FILE), DEFAULT_CONFIG_FILE), "Config file")
    ("graphfile", boost::program_options::value<std::string>(), "Graph file")
    ("featuresfile", boost::program_options::value<std::string>(), "Features file")

    ("undirected", boost::program_options::value<unsigned>()->default_value(unsigned(ZERO), ZERO_STR), "Graph type")
    ("dthreads", boost::program_options::value<unsigned>()->default_value(unsigned(NUM_DATA_THREADS), NUM_DATA_THREADS_STR), "Number of data threads")
    ("cthreads", boost::program_options::value<unsigned>()->default_value(unsigned(NUM_COMP_THREADS), NUM_COMP_THREADS_STR), "Number of compute threads")
    ("pofrequency", boost::program_options::value<unsigned>()->default_value(unsigned(PUSHOUT_FREQUENCY), PUSHOUT_FREQUENCY_STR), "Frequency of pushouts")
    ("dport", boost::program_options::value<unsigned>()->default_value(unsigned(DATA_PORT), DATA_PORT_STR), "Port for data communication")
    ("cport", boost::program_options::value<unsigned>()->default_value(unsigned(CONTROL_PORT_START), CONTROL_PORT_START_STR), "Port start for control communication")

    ("baseedges", boost::program_options::value<unsigned>()->default_value(unsigned(ZERO), ZERO_STR), "Percentage of edges in base graph")
    ("numbatches", boost::program_options::value<unsigned>()->default_value(unsigned(ZERO), ZERO_STR), "Number of mini-batches") 
    ("batchsize", boost::program_options::value<unsigned>()->default_value(unsigned(ZERO), ZERO_STR), "Size of mini-batches")
    ("deletepercent", boost::program_options::value<unsigned>()->default_value(unsigned(ZERO), ZERO_STR), "Deletion percent in mini-batches")
    ;

  boost::program_options::variables_map vm;

  boost::program_options::store(boost::program_options::command_line_parser(argc, argv).options(desc).allow_unregistered().run(), vm);
  boost::program_options::notify(vm);

  assert(vm.count("config"));

  std::string cFile = vm["config"].as<std::string>();

  std::ifstream cStream;
  cStream.open(cFile.c_str());

  boost::program_options::store(boost::program_options::parse_config_file(cStream, desc, true), vm);
  boost::program_options::notify(vm);

  if (vm.count("help")) {
    std::cout << desc << std::endl;
    exit(-1);
  }

  assert(vm.count("dthreads"));
  dThreads = vm["dthreads"].as<unsigned>();

  dThreads = 1; // forced to 1

  assert(vm.count("cthreads"));
  cThreads = vm["cthreads"].as<unsigned>();

  assert(vm.count("graphfile"));
  graphFile = vm["graphfile"].as<std::string>();

  assert(vm.count("featuresfile"));
  featuresFile = vm["featuresfile"].as<std::string>();

  assert(vm.count("undirected"));
  unsigned undir = vm["undirected"].as<unsigned>();
  undirected = (undir == 0) ? false : true;

  assert(vm.count("dport"));
  CommManager::setDataPort(vm["dport"].as<unsigned>());

  assert(vm.count("cport"));
  CommManager::setControlPortStart(vm["cport"].as<unsigned>());

  assert(vm.count("baseedges"));
  baseEdges = vm["baseedges"].as<unsigned>();

  assert(baseEdges <= 100);

  assert(vm.count("numbatches"));
  numBatches = vm["numbatches"].as<unsigned>();

  assert(vm.count("batchsize"));
  batchSize = vm["batchsize"].as<unsigned>();

  assert(vm.count("deletepercent"));
  deletePercent = vm["deletepercent"].as<unsigned>();

  fprintf(stderr, "config set to %s\n", cFile.c_str());
  fprintf(stderr, "dThreads set to %u\n", dThreads);
  fprintf(stderr, "cThreads set to %u\n", cThreads);
  fprintf(stderr, "graphFile set to %s\n", graphFile.c_str());
  fprintf(stderr, "featuresFile set to %s\n", featuresFile.c_str());
  fprintf(stderr, "undirected set to %s\n", undirected ? "true" : "false");
  fprintf(stderr, "pofrequency set to %u\n", poFrequency);

  fprintf(stderr, "baseEdges (percent) set to %u\n", baseEdges);
  fprintf(stderr, "numBatches set to %u\n", numBatches);
  fprintf(stderr, "batchSize set to %u\n", batchSize);
  fprintf(stderr, "deletePercent set to %u\n", deletePercent);
}

//shen
template <typename VertexType, typename EdgeType>
int Engine<VertexType, EdgeType>::readFeaturesFile(const std::string& fileName) {
  std::ifstream in;//Open file stream
  in.open(fileName);
  if(!in.good()) {
      fprintf(stderr, "Cannot open feature file: %s\n", fileName.c_str());
      return 1;
  }

  static const std::size_t LINE_BUFFER_SIZE=8192;
  char buffer[LINE_BUFFER_SIZE];

  std::vector<VertexType> feature_mat;
  //process each line
  while (in.eof()!=true){
      in.getline(buffer,LINE_BUFFER_SIZE);
      std::vector<std::string> splited_strings;
      VertexType feature_vec=VertexType();
      std::string line(buffer);
      //split each line into numbers
      boost::split(splited_strings, line, boost::is_any_of(", "), boost::token_compress_on);

      for (auto it=splited_strings.begin();it!=splited_strings.end();it++) {
          if(it->data()[0]!=0) //check null char at the end
              feature_vec.push_back(std::stof(it->data()));
      }
      feature_mat.push_back(feature_vec);
  }

  for(std::size_t i = 0;i<feature_mat.size();++i){
    //is ghost node
    auto git=graph.ghostVertices.find(i);
    if (git != graph.ghostVertices.end()){
      graph.updateGhostVertex(i,&feature_mat[i]);
      continue;
    }
    //is local node
    auto lit=graph.globalToLocalId.find(i);
    if (lit != graph.globalToLocalId.end()){
      graph.vertices[lit->second].setData(feature_mat[i]);
      continue;
    }
  }

  return 0;

}

template <typename VertexType, typename EdgeType>
void Engine<VertexType, EdgeType>::readPartsFile(std::string& partsFileName, Graph<VertexType, EdgeType>& lGraph) {
  std::ifstream infile(partsFileName.c_str());
  if(!infile.good())
    fprintf(stderr, "Cannot open patition file: %s\n", partsFileName.c_str());

  assert(infile.good());
  fprintf(stderr, "Reading patition file: %s\n", partsFileName.c_str());


  short partId;
  IdType lvid = 0; IdType gvid = 0;

  std::string line;
  while(std::getline(infile, line)) {
    if(line.size() == 0 || (line[0] < '0' || line[0] > '9'))
      continue;

    std::istringstream iss(line);
    if(!(iss >> partId))
      break;

    lGraph.vertexPartitionIds.push_back(partId);

    if(partId == nodeId) {
      lGraph.localToGlobalId[lvid] = gvid;
      lGraph.globalToLocalId[gvid] = lvid;

      ++lvid;
    }
    ++gvid;
  }

  lGraph.numGlobalVertices = gvid;
  lGraph.numLocalVertices = lvid;
}

template <typename VertexType, typename EdgeType>
void Engine<VertexType, EdgeType>::initGraph(Graph<VertexType, EdgeType>& lGraph) {
  lGraph.vertices.resize(lGraph.numLocalVertices);

  for(IdType i=0; i<lGraph.numLocalVertices; ++i) {
    lGraph.vertices[i].localIdx = i;
    lGraph.vertices[i].globalIdx = lGraph.localToGlobalId[i];
    lGraph.vertices[i].vertexLocation = INTERNAL_VERTEX;
    lGraph.vertices[i].vertexData.clear();
    lGraph.vertices[i].vertexData.push_back(defaultVertex);
    lGraph.vertices[i].graph = &lGraph;
  }
}

template <typename VertexType, typename EdgeType>
void Engine<VertexType, EdgeType>::processEdge(IdType& from, IdType& to, Graph<VertexType, EdgeType>& lGraph, std::set<IdType>* inTopics, std::set<IdType>* oTopics, bool streaming) {
  if(lGraph.vertexPartitionIds[from] == nodeId) {
    IdType lFromId = lGraph.globalToLocalId[from];
    IdType toId;
    EdgeLocationType eLocation;

    if(lGraph.vertexPartitionIds[to] == nodeId) {
      toId = lGraph.globalToLocalId[to];
      eLocation = LOCAL_EDGE_TYPE;
    } else {
      toId = to;
      eLocation = REMOTE_EDGE_TYPE;
      lGraph.vertices[lFromId].vertexLocation = BOUNDARY_VERTEX;
      if(oTopics != NULL) oTopics->insert(from);
      if(streaming) {
        CommManager::dataPushOut(from, (void*) lGraph.vertices[lFromId].data().data(), lGraph.vertices[lFromId].data().size() * sizeof(FeatType));
      }
    }

    if(edgeWeight != NULL)
      lGraph.vertices[lFromId].outEdges.push_back(OutEdge<EdgeType>(toId, eLocation, edgeWeight(from, to)));
    else
      lGraph.vertices[lFromId].outEdges.push_back(OutEdge<EdgeType>(toId, eLocation, defaultEdge));
  }

  if(lGraph.vertexPartitionIds[to] == nodeId) {
    IdType lToId = lGraph.globalToLocalId[to];
    IdType fromId;
    EdgeLocationType eLocation;

    if(lGraph.vertexPartitionIds[from] == nodeId) {
      fromId = lGraph.globalToLocalId[from];
      eLocation = LOCAL_EDGE_TYPE;
    } else {
      fromId = from;
      eLocation = REMOTE_EDGE_TYPE;

      typename std::map< IdType, GhostVertex<VertexType> >::iterator gvit = lGraph.ghostVertices.find(from);
      if(gvit == lGraph.ghostVertices.end()) {
        lGraph.ghostVertices[from] = GhostVertex<VertexType>(defaultVertex);

        gvit = lGraph.ghostVertices.find(from);
      }
      gvit->second.outEdges.push_back(lToId);

      if(inTopics != NULL) inTopics->insert(from);
    }

    if(edgeWeight != NULL)
      lGraph.vertices[lToId].inEdges.push_back(InEdge<EdgeType>(fromId, eLocation, edgeWeight(from, to)));
    else
      lGraph.vertices[lToId].inEdges.push_back(InEdge<EdgeType>(fromId, eLocation, defaultEdge));
  }
}

template <typename VertexType, typename EdgeType>
void Engine<VertexType, EdgeType>::receiveNewGhostValues(std::set<IdType>& inTopics) {
  while(inTopics.size() > 0) {
    IdType vid; VertexType value;
    CommManager::dataSyncPullIn(vid, value);
    if(inTopics.find(vid) != inTopics.end()) {
      typename std::map<IdType, GhostVertex<VertexType> >::iterator gvit = graph.ghostVertices.find(vid);
      assert(gvit != graph.ghostVertices.end());
      gvit->second.addData(&value); 
      inTopics.erase(vid);
    }
  }
  CommManager::flushDataControl();
}

template <typename VertexType, typename EdgeType>
void Engine<VertexType, EdgeType>::readGraphBS(std::string& fileName, std::set<IdType>& inTopics, std::vector<IdType>& outTopics) {
  // Read the partition file
  std::string partsFileName = fileName + PARTS_EXT;
  readPartsFile(partsFileName, graph);

  initGraph(graph);
  std::set<IdType> oTopics;

  // Read the edge file
  std::string edgeFileName = fileName + EDGES_EXT;
  std::ifstream infile(edgeFileName.c_str(), std::ios::binary);
  if(!infile.good())
    fprintf(stderr, "Cannot open BinarySnap file: %s\n", edgeFileName.c_str());

  assert(infile.good());
  fprintf(stderr, "Reading BinarySnap file: %s\n", edgeFileName.c_str());

  BSHeaderType<IdType> bSHeader;

  infile.read((char*) &bSHeader, sizeof(bSHeader));

  assert(bSHeader.sizeOfVertexType == sizeof(IdType));
  unsigned long long originalEdges = (unsigned long long) ((baseEdges / 100.0) * bSHeader.numEdges);

  graph.numGlobalEdges = 0;
  IdType srcdst[2];
  while(infile.read((char*) srcdst, bSHeader.sizeOfVertexType * 2)) {
    if(srcdst[0] == srcdst[1])
      continue;

    if(graph.numGlobalEdges < originalEdges) {
      processEdge(srcdst[0], srcdst[1], graph, &inTopics, &oTopics, false);
      if(undirected)
        processEdge(srcdst[1], srcdst[0], graph, &inTopics, &oTopics, false);
      ++graph.numGlobalEdges;
    } else {
      if((graph.vertexPartitionIds[srcdst[0]] == nodeId) || (graph.vertexPartitionIds[srcdst[1]] == nodeId)) {
        insertStream.push_back(std::make_tuple(globalInsertStreamSize, srcdst[0], srcdst[1]));
      }
      ++globalInsertStreamSize;
      if(globalInsertStreamSize > numBatches * batchSize)
        break;
    }
  }

  infile.close();

  findGhostDegrees(edgeFileName);
  setEdgeNormalizations();

  typename std::set<IdType>::iterator it;
  for(it = oTopics.begin(); it != oTopics.end(); ++it)
    outTopics.push_back(*it);
}

// Finds the in degree of the ghost vertices
template <typename VertexType, typename EdgeType>
void Engine<VertexType, EdgeType>::findGhostDegrees(std::string& fileName) {
	std::ifstream infile(fileName.c_str(), std::ios::binary);
	if (!infile.good()) {
		fprintf(stderr, "Cannot open BinarySnap file: %s\n", fileName.c_str());
	}

	assert(infile.good());
	fprintf(stderr, "Calculating the degree of ghost vertices\n");

	BSHeaderType<IdType> bsHeader;
	infile.read((char*) &bsHeader, sizeof(bsHeader));

	IdType srcdst[2];
	while (infile.read((char*) srcdst, bsHeader.sizeOfVertexType * 2)) {
		if (srcdst[0] == srcdst[1]) continue;

		typename std::map< IdType, GhostVertex<VertexType> >::iterator gvit = graph.ghostVertices.find(srcdst[1]);
		if (gvit != graph.ghostVertices.end()) {
			(gvit->second).incrementDegree();
		}	
	}
	
	infile.close();
}

// Sets the normalization factors on all edges
template<typename VertexType, typename EdgeType>
void Engine<VertexType, EdgeType>::setEdgeNormalizations() {
	for (Vertex<VertexType, EdgeType>& vertex: graph.vertices) {
		unsigned dstDeg = vertex.numInEdges() + 1;
		float dstNorm = std::pow(dstDeg, -.5);
		for (InEdge<EdgeType>& e: vertex.inEdges) {
			IdType vid = e.sourceId();
			if (e.getEdgeLocation() == LOCAL_EDGE_TYPE) {
				unsigned srcDeg = graph.vertices[vid].numInEdges() + 1;
				float srcNorm = std::pow(srcDeg, -.5);
				e.setData(srcNorm * dstNorm);
			} else {
				unsigned ghostDeg = graph.ghostVertices[vid].degree + 1;
				float ghostNorm = std::pow(ghostDeg, -.5);
				e.setData(ghostNorm * dstNorm);
			}
		}
	}
}

template <typename VertexType, typename EdgeType>
void Engine<VertexType, EdgeType>::readDeletionStream(std::string& fileName) {
  // Read the edge file
  std::string edgeFileName = fileName + DELS_EXT;
  std::ifstream infile(edgeFileName.c_str(), std::ios::binary);
  if(!infile.good())
    fprintf(stderr, "Cannot open BinarySnap file: %s\n", edgeFileName.c_str());

  assert(infile.good());
  fprintf(stderr, "Reading BinarySnap file: %s\n", edgeFileName.c_str());

  BSHeaderType<IdType> bSHeader;

  infile.read((char*) &bSHeader, sizeof(bSHeader));

  assert(bSHeader.sizeOfVertexType == sizeof(IdType));

  IdType srcdst[2];
  while(infile.read((char*) srcdst, bSHeader.sizeOfVertexType * 2)) {
    if(srcdst[0] == srcdst[1])
      continue;

    if((graph.vertexPartitionIds[srcdst[0]] == nodeId) || (graph.vertexPartitionIds[srcdst[1]] == nodeId)) {
      deleteStream.push_back(std::make_tuple(globalDeleteStreamSize, srcdst[0], srcdst[1]));
    }
    ++globalDeleteStreamSize;

    if(globalDeleteStreamSize > numBatches * (batchSize * (deletePercent / 100.0 + 1)))
      break;
  }

  infile.close();
}


/**
 *
 * Initialize the engine.
 * 
 */
template <typename VertexType, typename EdgeType>
void Engine<VertexType, EdgeType>::init(int argc, char* argv[], VertexType dVertex, EdgeType dEdge, EdgeType (*eWeight) (IdType, IdType)) {
  timInit = -getTimer();
  parseArgs(argc, argv);
  
  NodeManager::init(ZKHOST_FILE, HOST_FILE);

  CommManager::init();

  nodeId = NodeManager::getNodeId();
  numNodes = NodeManager::getNumNodes();

  assert(numNodes <= 256); // ASSUMPTION IS THAT NODES ARE NOT MORE THAN 256.

  std::set<IdType> inTopics;
  std::vector<IdType> outTopics;
  inTopics.insert(IAMDONE);
  inTopics.insert(IAMNOTDONE);
  inTopics.insert(ITHINKIAMDONE); 
  outTopics.push_back(IAMDONE);
  outTopics.push_back(IAMNOTDONE);
  outTopics.push_back(ITHINKIAMDONE); 

  CommManager::subscribeData(&inTopics, &outTopics);
  inTopics.clear();
  outTopics.clear();

  defaultVertex = dVertex;
  defaultEdge = dEdge;
  edgeWeight = eWeight;

  //fprintf(stderr, "defaultVertex = %.3lf\n", defaultVertex);
  readGraphBS(graphFile, inTopics, outTopics);
  graph.printGraphMetrics();
  fprintf(stderr, "Insert Stream size = %zd\n", insertStream.size());

  readFeaturesFile(featuresFile); //input filename is hardcoded here

  if (numBatches != 0) {
    readDeletionStream(graphFile);
    fprintf(stderr, "Delete Stream size = %zd\n", deleteStream.size());
  }

  engineContext.setNumVertices(graph.numGlobalVertices);

  //CommManager::subscribeData(&inTopics, &outTopics);

  scheduler = new BitsetScheduler(graph.numLocalVertices);
  engineContext.setScheduler(scheduler);

  shadowScheduler = new DenseBitset(graph.numLocalVertices);
  trimScheduler = new DenseBitset(graph.numLocalVertices);

  lockCurrId.init();

  pthread_mutex_init(&mtxCompWaiter, NULL);
  pthread_cond_init(&condCompWaiter, NULL);

  pthread_mutex_init(&mtxDataWaiter, NULL);
  pthread_cond_init(&condDataWaiter, NULL);

  barComp.init(cThreads);
  barCompData.init(2);                            // master cthread and datacommunicator

  pthread_barrier_init(&barDebug, NULL, cThreads + 10);

  computePool = new ThreadPool(cThreads);
  computePool->createPool();

  CommManager::flushDataControl();
  dataPool = new ThreadPool(dThreads);
  dataPool->createPool();

  graph.compactGraph();

  timInit += getTimer();
  fprintf(stderr, "Init complete\n"); 
  NodeManager::barrier("init");
}

template <typename VertexType, typename EdgeType>
void Engine<VertexType, EdgeType>::destroy() {
  NodeManager::destroy();
  CommManager::destroy();
  computePool->destroyPool();
  //fprintf(stderr, "H0\n");
  //dataPool->sync();
  dataPool->destroyPool();
  //fprintf(stderr, "H1\n");
}

template <typename VertexType, typename EdgeType>
IdType Engine<VertexType, EdgeType>::numVertices() {
  return graph.numGlobalVertices;
}

template <typename VertexType, typename EdgeType>
bool Engine<VertexType, EdgeType>::master() {
  return NodeManager::amIMaster();
}


template <typename VertexType, typename EdgeType>
void Engine<VertexType, EdgeType>::signalAll() {
  scheduler->scheduleAll();
}

template <typename VertexType, typename EdgeType>
void Engine<VertexType, EdgeType>::signalVertex(IdType vId) {
  if(graph.vertexPartitionIds[vId] == nodeId) {
    scheduler->schedule(graph.globalToLocalId[vId]);
  }
}

template <typename VertexType, typename EdgeType>
void Engine<VertexType, EdgeType>::shadowSignalVertex(IdType vId) {
  //assert(graph.vertexPartitionIds[vId] == nodeId);  // ths is always true
  shadowScheduler->setBit(graph.globalToLocalId[vId]);
}

template <typename VertexType, typename EdgeType>
void Engine<VertexType, EdgeType>::shadowUnsignalVertex(IdType vId) {
  //assert(graph.vertexPartitionIds[vId] == nodeId);  // ths is always true
  shadowScheduler->clearBit(graph.globalToLocalId[vId]);
}

template <typename VertexType, typename EdgeType>
void Engine<VertexType, EdgeType>::trimmed(IdType vId) {
  trimScheduler->setBit(graph.globalToLocalId[vId]);
}

template <typename VertexType, typename EdgeType>
void Engine<VertexType, EdgeType>::notTrimmed(IdType vId) {
  trimScheduler->clearBit(graph.globalToLocalId[vId]);
}

template <typename VertexType, typename EdgeType>
void  Engine<VertexType, EdgeType>::printEngineMetrics() {
  fprintf(stderr, "EngineMetrics: Init time: %.3lf ms\n", timInit);
  fprintf(stderr, "EngineMetrics: Processing time: %.3lf ms\n", allTimProcess);
}

template <typename VertexType, typename EdgeType>
void  Engine<VertexType, EdgeType>::activateEndPoints(IdType from, IdType to, InOutType io, void (*oadHandler) (VertexType& v)) {
  switch(io) {

    case SRC:
      signalVertex(from);
      shadowSignalVertex(from);
      if(graph.vertexPartitionIds[from] == nodeId)
        oadHandler(graph.vertices[graph.globalToLocalId[from]].data());
      break;

    case DST:
      signalVertex(to);
      shadowSignalVertex(to);
      if(graph.vertexPartitionIds[to] == nodeId)
        oadHandler(graph.vertices[graph.globalToLocalId[to]].data());
      break;

    case BOTH:
      signalVertex(from);
      shadowSignalVertex(from);
      if(graph.vertexPartitionIds[from] == nodeId) 
        oadHandler(graph.vertices[graph.globalToLocalId[from]].data());
      signalVertex(to);
      shadowSignalVertex(to);
      if(graph.vertexPartitionIds[to] == nodeId)
        oadHandler(graph.vertices[graph.globalToLocalId[to]].data());
      break;

    case NEITHER:
      break;

    default:
      assert(false);

  }
}

template <typename VertexType, typename EdgeType>
void  Engine<VertexType, EdgeType>::activateEndPoints2(IdType from, IdType to, InOutType io, void (*oadHandler) (VertexType& v, LightEdge<VertexType, EdgeType>& edge), LightEdge<VertexType, EdgeType>& edge) {
  switch(io) {

    case SRC:
      assert(false);
      signalVertex(from);
      shadowSignalVertex(from);
      if(graph.vertexPartitionIds[from] == nodeId)
        oadHandler(graph.vertices[graph.globalToLocalId[from]].data(), edge);
      break;

    case DST:
      signalVertex(to);
      shadowSignalVertex(to);
      if(graph.vertexPartitionIds[to] == nodeId)
        oadHandler(graph.vertices[graph.globalToLocalId[to]].data(), edge);
      break;

    case BOTH:
      assert(false);
      signalVertex(from);
      shadowSignalVertex(from);
      if(graph.vertexPartitionIds[from] == nodeId) 
        oadHandler(graph.vertices[graph.globalToLocalId[from]].data(), edge);
      signalVertex(to);
      shadowSignalVertex(to);
      if(graph.vertexPartitionIds[to] == nodeId)
        oadHandler(graph.vertices[graph.globalToLocalId[to]].data(), edge);
      break;

    case NEITHER:
      assert(false);
      break;

    default:
      assert(false);

  }
}

template <typename VertexType, typename EdgeType>
void  Engine<VertexType, EdgeType>::addEdge(IdType from, IdType to, std::set<IdType>* inTopics) { // globalIds
  processEdge(from, to, graph, inTopics, NULL, true);
}

template <typename VertexType, typename EdgeType>
void  Engine<VertexType, EdgeType>::deleteEdge(IdType from, IdType to) { // globalIds
  if(graph.vertexPartitionIds[from] == nodeId) {
    IdType lFromId = graph.globalToLocalId[from];
    IdType toId = (graph.vertexPartitionIds[to] == nodeId ? graph.globalToLocalId[to] : to);

    typename std::vector<OutEdge<EdgeType> >::iterator it;
    for(it = graph.vertices[lFromId].outEdges.begin(); it != graph.vertices[lFromId].outEdges.end(); ++it) {
      if(it->destId() == toId) {
        graph.vertices[lFromId].outEdges.erase(it);
        break;
      }
    }
  }

  if(graph.vertexPartitionIds[to] == nodeId) {
    IdType lToId = graph.globalToLocalId[to];
    bool remoteEdge = (graph.vertexPartitionIds[from] != nodeId);
    IdType fromId = (remoteEdge ? from : graph.globalToLocalId[from]);

    typename std::vector<InEdge<EdgeType> >::iterator it;
    for(it = graph.vertices[lToId].inEdges.begin(); it != graph.vertices[lToId].inEdges.end(); ++it) {
      if(it->sourceId() == fromId) {
        graph.vertices[lToId].inEdges.erase(it);
        break;
      }
    }

    if(remoteEdge) {
      typename std::map<IdType, GhostVertex<VertexType> >::iterator gvit = graph.ghostVertices.find(from);
      assert(gvit != graph.ghostVertices.end());
      typename  std::vector<IdType>::iterator it;
      for(it = gvit->second.outEdges.begin(); it != gvit->second.outEdges.end(); ++it) {
        if(*it == lToId) {
          gvit->second.outEdges.erase(it);
          break; 
        }
      }
    }
  }
}

template <typename VertexType, typename EdgeType>
LightEdge<VertexType, EdgeType> Engine<VertexType, EdgeType>::deleteEdge2(IdType from, IdType to) { // globalIds
  LightEdge<VertexType, EdgeType> retEdge;
  retEdge.fromId = from; retEdge.toId = to; retEdge.valid = false;

  if(graph.vertexPartitionIds[from] == nodeId) {
    IdType lFromId = graph.globalToLocalId[from];
    IdType toId = (graph.vertexPartitionIds[to] == nodeId ? graph.globalToLocalId[to] : to);

    typename std::vector<OutEdge<EdgeType> >::iterator it;
    for(it = graph.vertices[lFromId].outEdges.begin(); it != graph.vertices[lFromId].outEdges.end(); ++it) {
      if(it->destId() == toId) {
        graph.vertices[lFromId].outEdges.erase(it);
        break;
      }
    }
  }

  if(graph.vertexPartitionIds[to] == nodeId) {
    IdType lToId = graph.globalToLocalId[to];
    bool remoteEdge = (graph.vertexPartitionIds[from] != nodeId);
    IdType fromId = (remoteEdge ? from : graph.globalToLocalId[from]);

    typename std::vector<InEdge<EdgeType> >::iterator it;
    unsigned ct = 0;
    for(it = graph.vertices[lToId].inEdges.begin(); it != graph.vertices[lToId].inEdges.end(); ++it) {
      if(it->sourceId() == fromId) {
        retEdge.to = graph.vertices[lToId].data();
        retEdge.from = graph.vertices[lToId].getSourceVertexData(ct);
        retEdge.edge = graph.vertices[lToId].getInEdgeData(ct);
        retEdge.valid = true; 
        graph.vertices[lToId].inEdges.erase(it);
        break;
      }
      ++ct;
    }

    if(remoteEdge) {
      typename std::map<IdType, GhostVertex<VertexType> >::iterator gvit = graph.ghostVertices.find(from);
      assert(gvit != graph.ghostVertices.end());
      typename  std::vector<IdType>::iterator it;
      for(it = gvit->second.outEdges.begin(); it != gvit->second.outEdges.end(); ++it) {
        if(*it == lToId) {
          gvit->second.outEdges.erase(it);
          break; 
        }
      }
    }
  }

  return retEdge;
}

template <typename VertexType, typename EdgeType>
bool Engine<VertexType, EdgeType>::deleteEdge3(IdType from, IdType to) { // globalIds
  bool treeEdge = false;

  if(graph.vertexPartitionIds[from] == nodeId) {
    IdType lFromId = graph.globalToLocalId[from];
    IdType toId = (graph.vertexPartitionIds[to] == nodeId ? graph.globalToLocalId[to] : to);

    typename std::vector<OutEdge<EdgeType> >::iterator it;
    for(it = graph.vertices[lFromId].outEdges.begin(); it != graph.vertices[lFromId].outEdges.end(); ++it) {
      if(it->destId() == toId) {
        graph.vertices[lFromId].outEdges.erase(it);
        break;
      }
    }
  }

  if(graph.vertexPartitionIds[to] == nodeId) {
    IdType lToId = graph.globalToLocalId[to];
    bool remoteEdge = (graph.vertexPartitionIds[from] != nodeId);
    IdType fromId = (remoteEdge ? from : graph.globalToLocalId[from]);

    treeEdge = (graph.vertices[lToId].parent() == from);

    typename std::vector<InEdge<EdgeType> >::iterator it;
    unsigned ct = 0;
    for(it = graph.vertices[lToId].inEdges.begin(); it != graph.vertices[lToId].inEdges.end(); ++it) {
      if(it->sourceId() == fromId) {
        graph.vertices[lToId].inEdges.erase(it);
        break;
      }
      ++ct;
    }

    if(remoteEdge) {
      typename std::map<IdType, GhostVertex<VertexType> >::iterator gvit = graph.ghostVertices.find(from);
      assert(gvit != graph.ghostVertices.end());
      typename  std::vector<IdType>::iterator it;
      for(it = gvit->second.outEdges.begin(); it != gvit->second.outEdges.end(); ++it) {
        if(*it == lToId) {
          gvit->second.outEdges.erase(it);
          break; 
        }
      }
    }
  }

  return treeEdge;
}

template <typename VertexType, typename EdgeType>
void Engine<VertexType, EdgeType>::setOnAddDelete(InOutType oa, void (*oaH) (VertexType& v), InOutType od, void (*odH) (VertexType& v)) {
  onAdd = oa;
  onAddHandler = oaH;

  onDelete = od;
  onDeleteHandler = odH;
}

template <typename VertexType, typename EdgeType>
void Engine<VertexType, EdgeType>::setOnDeleteSmartHandler(void (*odSmartHandler) (VertexType& v, LightEdge<VertexType, EdgeType>& e)) {
  onDeleteSmartHandler = odSmartHandler; 
}

template <typename VertexType, typename EdgeType>
void Engine<VertexType, EdgeType>::run(VertexProgram<VertexType, EdgeType>* vProgram, bool printEM) {
  NodeManager::barrier("run"); 
  fprintf(stderr, "Engine::run called.\n");
  timProcess = -getTimer();

  NodeManager::startProcessing();
  vertexProgram = vProgram;
  currId = 0; iteration = 0;
  firstIteration = true; compDone = false; die = false; 
  halt = false;
  scheduler->newIteration();

  fprintf(stderr, "Starting data communicator\n");
  dataPool->perform(dataCommunicator);
  fprintf(stderr, "Starting worker task\n");
  computePool->perform(worker);
  computePool->sync();

  timProcess += getTimer();

  dataPool->sync();

  fprintf(stderr, "Processing completed on node %u in %u iterations\n", nodeId, iteration);
  if(master() && printEM)
    printEngineMetrics();
}

template <typename VertexType, typename EdgeType>
void Engine<VertexType, EdgeType>::quickRun(VertexProgram<VertexType, EdgeType>* vProgram, bool metrics) {
  NodeManager::barrier("quickrun");
  fprintf(stderr, "Engine::quickRun called.\n");
  vertexProgram = vProgram;

  if(metrics) timProcess = -getTimer();

  currId = 0; iteration = 0;
  firstIteration = true; compDone = false; die = false; 
  halt = false;
  scheduler->newIteration();

  dataPool->perform(dataCommunicator);
  computePool->perform(worker);
  computePool->sync();

  if(metrics) timProcess += getTimer();

  dataPool->sync();

  //fprintf(stderr, "quickRun completed on node %u in %u iterations took %.3lf ms\n", nodeId, iteration, timProcess);
}


template <typename VertexType, typename EdgeType>
void Engine<VertexType, EdgeType>::processAll(VertexProgram<VertexType, EdgeType>* vProgram) {
  vProgram->beforeIteration(engineContext);

  for(IdType i=0; i<graph.numLocalVertices; ++i)
    vProgram->processVertex(graph.vertices[i]);

  vProgram->afterIteration(engineContext);
}


/**
 *
 * Major part of the engine's computation logic is done by workers. When the engine runs it wakes threads up from the thread pool
 * and assign a worker function for each.
 * 
 */
template <typename VertexType, typename EdgeType>
void Engine<VertexType, EdgeType>::worker(unsigned tid, void* args) {
  static bool compHalt = false;

  // Outer while loop. Looping infinitely, looking for a new task to handle.
  while (1) {
    IdType vid;
    bool found = false;

    lockCurrId.lock();

    // Inner while loop. Looping until got a current vertex id to handle.
    while (1) {
      found = false;

      // All local vertices have been processed. Hit the barrier and wait for next iteration / decide to halt.
      if (currId >= graph.numLocalVertices) {

        // Non-master threads.
        if (tid != 0) {
          lockCurrId.unlock();

          //## Worker barrier 1: Everyone reach to this point, then only master will work. ##//
          barComp.wait();

          //## Worker barrier 2: Master has finished its checking work. ##//
          barComp.wait();

          // Master thread decides to halt, and the communicator confirms it. So going to die.
          if (compHalt)
            break;

          // New iteration starts.
          lockCurrId.lock();
          continue;

        // Master thread (tid == 0).
        } else {
          lockCurrId.unlock();

          //## Worker barrier 1: Everyone reach to this point, then only master will work. ##//
          barComp.wait();

          if (!firstIteration) {
            ++iteration;
            gotoNextLayer();
          } else
            firstIteration = false;

          // Yes there are further scheduled vertices. Please start a new iteration.
          bool hltBool = ((timProcess + getTimer() > 500) && !(scheduler->anyScheduledTasks()));  // After 1 sec only!
          if (!hltBool) {
            scheduler->newIteration();
            currId = 0;       // This is unprotected by lockCurrId because only master executes.
            lockCurrId.lock();

            //## Worker barrier 2: Starting a new iteration. ##//
            fprintf(stderr, "Starting a new iteration %u at %.3lf ms...\n", iteration, timProcess + getTimer());
            barComp.wait();

          // No more, so deciding to halt. But still needs the communicator to check if there will be further scheduling invoked by ghost
          // vertices. If so we are stilling going to the next iteration.
          } else {

            //## Global Comm barrier 1: Wait for every node's communicator to enter the decision making procedure. ##//
            fprintf(stderr, "Deciding to halt at iteration %u...\n", iteration);
            NodeManager::barrier(COMM_BARRIER);

            pthread_mutex_lock(&mtxDataWaiter);
            compDone = true;    // Set this to true, so the communicator will start the finish-up checking procedure.
            pthread_mutex_unlock(&mtxDataWaiter);

            //## Worker-Comm barrier 1: Wake up dataCommunicator to enter the finish-up checking procedure. ##//
            barCompData.wait();

            //## Worker-Comm barrier 2: Wait for dataCommunicator's decision on truely halt / not. ##//
            barCompData.wait();

            pthread_mutex_lock(&mtxDataWaiter);
            compDone = false;
            compHalt = halt;
            die = halt;
            pthread_mutex_unlock(&mtxDataWaiter);

            //## Worker-Comm barrier 3: Worker has set compDone = false. ##//
            barCompData.wait();

            //## Worker-Comm barrier 4: Wait for communicator to ensure compDone == false. ##//
            barCompData.wait();

            // Really going to halt.
            if (compHalt) {

              //## Worker barrier 2: Communicator also decides to halt. So going to die. ##//
              fprintf(stderr, "Communicator confirms the halt at iteration %u.\n", iteration);
              barComp.wait();

              break;

            // Communicator denies the halt. Continue working.
            } else {

              if (scheduler->anyScheduledTasks()) {  // Not always true. Communicator may decide to continue because of another machine is not done. In this situation,
                                                     // I will not enter a new iteration, but rather hit the COMM_BARRIER again and wait for other machines to decide to halt again.
                scheduler->newIteration();
                currId = 0;   // This is unprotected by lockCurrId because only master executes.
              }

              lockCurrId.lock();

              //////
              // Send to lambda here ???
              //////

              //## Worker barrier 2: Communicator decides we cannot halt now. Continue working. ##//
              fprintf(stderr, "Communicator denies the halt at iteration %u.\n", iteration);
              barComp.wait();

              continue;
            }
          } // end of else: deciding to halt
        } // end of else: tid == 0
      } // end of if: currId >= graph.numLocalVertices

      // Do work on the current processing local vertex id.
      if (scheduler->isScheduled(currId)) {
        vid = currId;
        found = true;
        ++currId;
        break;
      } else
        ++currId;
    } // end of inner while loop

    lockCurrId.unlock();

    // Out of inner while loop but no vertex to handle is given.
    if (!found) {
      fprintf(stderr, ">- Worker %u has nothing to work on, hence leaving....\n", tid);
      break;
    }

    // I got a current vertex id in `vid` to handle now. Doing the task.
    Vertex<VertexType, EdgeType>& v = graph.vertices[vid];
    assert(scheduler->isScheduled(vid));

    if (!firstIteration)
      vertexProgram->update(v, engineContext);
    
    if (firstIteration | !reachOutputLayer()) {
      bool remoteScat = true;

      for (unsigned i = 0; i < v.numOutEdges(); ++i) {

        // A local edge. Schedule that neighbor immediately.
        if (v.getOutEdge(i).getEdgeLocation() == LOCAL_EDGE_TYPE)
          scheduler->schedule(v.getOutEdge(i).destId());

        // A remote edge. Send my vid to other machines, for them to update their ghost vertex's value and schedule its neighbors.
        else if (remoteScat) {
          CommManager::dataPushOut(graph.localToGlobalId[vid], (void*)v.data().data(), sizeof(FeatType) * v.data().size());
          remoteScat = false;   // Such send should only happen once, no matter how many remote edges this vertex has.
        }
      }
    }
  } // end of outer while loop
}

template <typename VertexType, typename EdgeType>
bool Engine<VertexType, EdgeType>::reachOutputLayer() {
  // Do not need mutex lock here, since will only be read but not written.
  return curr_layer >= 5;  // 5 here is just a meaningless example.
}

template <typename VertexType, typename EdgeType>
void Engine<VertexType, EdgeType>::gotoNextLayer() {
  // Do not need mutex lock here, since will only be called by master at iteration finish up.
  ++curr_layer;
}

template <typename VertexType, typename EdgeType>
void Engine<VertexType, EdgeType>::updateGhostVertex(IdType vid, VertexType& value) {
  graph.updateGhostVertex(vid, &value);
  std::vector<IdType>* outEdges = &(graph.ghostVertices[vid].outEdges);
  for(unsigned i=0; i<outEdges->size(); ++i)
    scheduler->schedule(outEdges->at(i));
}

template <typename VertexType, typename EdgeType>
void Engine<VertexType, EdgeType>::conditionalUpdateGhostVertex(IdType vid, VertexType& value) {
  if(graph.ghostVertices.find(vid) != graph.ghostVertices.end()) {
    graph.updateGhostVertex(vid, &value);
    std::vector<IdType>* outEdges = &(graph.ghostVertices[vid].outEdges);
    for(unsigned i=0; i<outEdges->size(); ++i)
      scheduler->schedule(outEdges->at(i));
  }
}


/**
 *
 * Major part of the engine's communication logic is done by data threads. These threads loop asynchronously with computation workers.
 * 
 */
template <typename VertexType, typename EdgeType>
void Engine<VertexType, EdgeType>::dataCommunicator(unsigned tid, void* args) {
  IdType vid;
  VertexType value;

  // While loop, looping infinitely to get the next message.
  while(1) {

    // Pull in the next message, but receives nothing.
    if (!CommManager::dataPullIn(vid, value)) {

      // No message now but computation workers haven't decided to halt yet, so trying to get the next message.
      if (!compDone)
        continue;

      // No message now and computation workers decided to halt. Going into the finish-up checking procedure.

      //## Global Datacomm barrier 1: Wait for every node to enter the finish-up procedure. ##//
      NodeManager::barrier(DATACOMM_BARRIER);

      //## Worker-Comm barrier 1: Going into the finish-up checking procedure. ##//
      barCompData.wait();

      pthread_mutex_lock(&mtxDataWaiter);
      assert(compDone);
      CommManager::dataPushOut(ITHINKIAMDONE, (void*)value.data(), value.size() * sizeof(FeatType));

      // Loop until receiving all nodes' respond on whether I should really halt / not.
      unsigned rem = numNodes;
      bool thinkHalt = !(scheduler->anyScheduledTasks());
      halt = true;
      while (rem > 0) {
        IdType mType;

        while (!CommManager::dataPullIn(mType, value)) { }   // Receive the next message.

        if (mType == ITHINKIAMDONE) {  // One also thinks he is done, just like me. Good.
          --rem;
        } else if (mType == IAMNOTDONE || mType == IAMDONE) {    // Impossible.
          assert(false);
        } else {  // There is a vertex id message on the way so someone will update its ghost vertices and schedule its neighbors.
                  // So not going to halt.
          vid = mType;
          conditionalUpdateGhostVertex(vid, value);
          thinkHalt = false;
        }
      }

      //## Global Datacomm barrier 2: Wait for all nodes to be ready for making the decision. ##//
      NodeManager::barrier(DATACOMM_BARRIER);

      // Make and push the decision of whether I am done / not.
      if (thinkHalt) {
        CommManager::dataPushOut(IAMDONE, (void*)value.data(), value.size() * sizeof(FeatType));
      } else {
        CommManager::dataPushOut(IAMNOTDONE, (void*)value.data(), value.size() * sizeof(FeatType));
        halt = false;
      }

      // Loop until receiving all nodes' decision on whether it is done / not.
      IdType mType;
      unsigned totalRem = numNodes;
      unsigned finalRem = numNodes;
      while (totalRem > 0) {

        while (!CommManager::dataPullIn(mType, value)) { }   // Receive the next message.

        if (mType == IAMDONE) {  // One truely decides to finish.
          --finalRem;
          --totalRem;
        } else if (mType == IAMNOTDONE) {  // One cannot finish, so not halting.
          halt = false;
          --totalRem;
        } else {  // Impossible.
          assert(false);
        }
      }

      //## Global Datacomm barrier 3: Required because one of the machines can go so far ahead that we may start receiving data from their worker threads. ##//
      NodeManager::barrier(DATACOMM_BARRIER);

      if (finalRem != 0)
        assert(halt == false); 
      pthread_mutex_unlock(&mtxDataWaiter);

      //## Worker-Comm barrier 2: The dataCommunicator made its decision on truely halt / not. ##//
      barCompData.wait();

      //## Worker-Comm barrier 3: Wait for worker to set compDone = false. ##//
      barCompData.wait();

      pthread_mutex_lock(&mtxDataWaiter);
      assert(compDone == false);
      pthread_mutex_unlock(&mtxDataWaiter);

      //## Worker-Comm barrier 4: Communicator ensures that compDone == false. ##//
      barCompData.wait();

      // Communicator confirms the halt, so going to die.
      if (halt) {
        fprintf(stderr, ">- Communicator %u confirms the halt, hence leaving...\n", tid);
        break;
      }

    // Pull in the next message, and recieves something. Process this message.
    } else {

      if (vid == IAMDONE || vid == IAMNOTDONE || vid == ITHINKIAMDONE) {   // Impossible.
        assert(false);
      } else {  // Receives a vid. Update the corresponding ghost vertex (if it is one of my ghost vertices), and schedule its neighbors.
        conditionalUpdateGhostVertex(vid, value);
      }
    }
  }
}

template <typename VertexType, typename EdgeType>
unsigned Engine<VertexType, EdgeType>::getPreviousAliveNodeId(unsigned nId) {
  unsigned ret = (nId == 0) ? numNodes - 1 : nId - 1;
  return ret;   
}

template <typename VertexType, typename EdgeType>
unsigned Engine<VertexType, EdgeType>::getNextAliveNodeId(unsigned nId) {
  unsigned ret = (nId + 1) % numNodes;
  return ret;
}

template <typename VertexType, typename EdgeType>
template <typename T>
T Engine<VertexType, EdgeType>::sillyReduce(T value, T (*reducer)(T, T)) {
  if(master()) {
    for(unsigned i=0; i<numNodes; ++i) {
      if(i == nodeId)
        continue;

      T v;
      CommManager::controlSyncPullIn(i, &v, sizeof(T));
      value = reducer(value, v);
    }
  } else {
    unsigned masterId = NodeManager::masterId();
    CommManager::controlPushOut(masterId, (void*) &value, sizeof(T));
  }
  return value;
}

#endif //__ENGINE_HPP__
