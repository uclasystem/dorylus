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
  fprintf(stderr, "undirected set to %s\n", undirected ? "true" : "false");
  fprintf(stderr, "pofrequency set to %u\n", poFrequency);

  fprintf(stderr, "baseEdges (percent) set to %u\n", baseEdges);
  fprintf(stderr, "numBatches set to %u\n", numBatches);
  fprintf(stderr, "batchSize set to %u\n", batchSize);
  fprintf(stderr, "deletePercent set to %u\n", deletePercent);
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
        CommManager::dataPushOut(from, (void*) lGraph.vertices[lFromId].vertexData.data(), lGraph.vertices[lFromId].vertexData.size() * sizeof(FeatType));
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
      gvit->second.setData(&value); 
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
  inTopics.insert(IAMDONE); inTopics.insert(IAMNOTDONE); inTopics.insert(ITHINKIAMDONE); 
  outTopics.push_back(IAMDONE); outTopics.push_back(IAMNOTDONE); outTopics.push_back(ITHINKIAMDONE); 

  for(IdType i=PUSHOUT_REQ_BEGIN; i<PUSHOUT_REQ_END; ++i) {
    inTopics.insert(i);
    outTopics.push_back(i);
  }

  for(IdType i=PUSHOUT_RESP_BEGIN; i<PUSHOUT_RESP_END; ++i) {
    inTopics.insert(i);
    outTopics.push_back(i);
  }


  CommManager::subscribeData(&inTopics, &outTopics);
  inTopics.clear(); outTopics.clear();

  defaultVertex = dVertex;
  defaultEdge = dEdge;
  edgeWeight = eWeight;

  //fprintf(stderr, "defaultVertex = %.3lf\n", defaultVertex);
  readGraphBS(graphFile, inTopics, outTopics);
  graph.printGraphMetrics();
  fprintf(stderr, "Insert Stream size = %zd\n", insertStream.size());

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
        oadHandler(graph.vertices[graph.globalToLocalId[from]].vertexData);
      break;

    case DST:
      signalVertex(to);
      shadowSignalVertex(to);
      if(graph.vertexPartitionIds[to] == nodeId)
        oadHandler(graph.vertices[graph.globalToLocalId[to]].vertexData);
      break;

    case BOTH:
      signalVertex(from);
      shadowSignalVertex(from);
      if(graph.vertexPartitionIds[from] == nodeId) 
        oadHandler(graph.vertices[graph.globalToLocalId[from]].vertexData);
      signalVertex(to);
      shadowSignalVertex(to);
      if(graph.vertexPartitionIds[to] == nodeId)
        oadHandler(graph.vertices[graph.globalToLocalId[to]].vertexData);
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
        oadHandler(graph.vertices[graph.globalToLocalId[from]].vertexData, edge);
      break;

    case DST:
      signalVertex(to);
      shadowSignalVertex(to);
      if(graph.vertexPartitionIds[to] == nodeId)
        oadHandler(graph.vertices[graph.globalToLocalId[to]].vertexData, edge);
      break;

    case BOTH:
      assert(false);
      signalVertex(from);
      shadowSignalVertex(from);
      if(graph.vertexPartitionIds[from] == nodeId) 
        oadHandler(graph.vertices[graph.globalToLocalId[from]].vertexData, edge);
      signalVertex(to);
      shadowSignalVertex(to);
      if(graph.vertexPartitionIds[to] == nodeId)
        oadHandler(graph.vertices[graph.globalToLocalId[to]].vertexData, edge);
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
        retEdge.to = graph.vertices[lToId].vertexData;
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
void Engine<VertexType, EdgeType>::streamRun(VertexProgram<VertexType, EdgeType>* vProgram, VertexProgram<VertexType, EdgeType>* wProgram, void (*reset)(void), bool printEM) {
  NodeManager::barrier("streamrun");
  fprintf(stderr, "Engine::streamRun called.\n");

  NodeManager::startProcessing();

  unsigned numDeletions = unsigned(batchSize * (deletePercent / 100.0));
  unsigned numInsertions = batchSize - numDeletions;

  unsigned long long currentGlobalInsertCounter = 0;
  unsigned long long currentGlobalDeleteCounter = 0;

  typename std::vector<std::tuple<unsigned long long, IdType, IdType> >::iterator insIter = insertStream.begin();
  typename std::vector<std::tuple<unsigned long long, IdType, IdType> >::iterator delIter = deleteStream.begin();

  unsigned batchNo = 0;
  engineContext.setCurrentBatch(batchNo);
  quickRun(vProgram, true);
  if(master()) fprintf(stderr, "Base Batch %u completed on node %u in %u iterations took %.3lf ms\n", batchNo, nodeId, iteration, timProcess);
  allTimProcess += timProcess;

  processAll(wProgram);

  while(batchNo++ < numBatches) {
    currentGlobalInsertCounter += numInsertions;
    currentGlobalDeleteCounter += numDeletions;

    unsigned trueAdditions = 0, trueDeletions = 0;

    // Insertions
    std::set<IdType> inTopics;
    while(insIter != insertStream.end()) {
      unsigned long long ts;
      IdType from, to; 
      std::tie(ts, from, to) = *insIter;
      if(ts < currentGlobalInsertCounter) {
        addEdge(from, to, &inTopics);
        activateEndPoints(from, to, onAdd, onAddHandler);
        if(undirected) {
          addEdge(to, from, &inTopics);
          activateEndPoints(to, from, onAdd, onAddHandler);
        }
        ++insIter;

        if(graph.vertexPartitionIds[to] == nodeId)
          ++trueAdditions;
        continue;
      }
      break;
    }

    receiveNewGhostValues(inTopics); 

    // Deletions
    while(delIter != deleteStream.end()) {
      unsigned long long ts;
      IdType from, to; 
      std::tie(ts, from, to) = *delIter;
      if(ts < currentGlobalDeleteCounter) {
        deleteEdge(from, to);
        activateEndPoints(from, to, onDelete, onDeleteHandler);
        if(undirected) {
          deleteEdge(to, from);
          activateEndPoints(to, from, onDelete, onDeleteHandler);
        }
        ++delIter;
        
        if(graph.vertexPartitionIds[to] == nodeId)
          ++trueDeletions;
        continue;
      }
      break;
    }

    graph.compactGraph();

    trueAdditions = sillyReduce(trueAdditions, &sumReducer);
    trueDeletions = sillyReduce(trueDeletions, &sumReducer);
    if(master()) fprintf(stderr, "Batch %u constructed using %u insertions and %u deletions\n", batchNo, trueAdditions, trueDeletions);

    double onlyBatchTime = 0.0;

    double tmr = 0.0;
    if(reset != NULL) {
      tmr = -getTimer();
      reset();
      tmr += getTimer();
      allTimProcess += tmr;
      onlyBatchTime += tmr;
    }
    if(master()) fprintf(stderr, "Batch %u phase 1 completed on node %u took %.3lf ms\n", batchNo, nodeId, tmr);

    //------

    engineContext.setCurrentBatch(batchNo);

    quickRun(vProgram, true);
    if(master()) fprintf(stderr, "Batch %u phase 2 completed on node %u in %u iterations took %.3lf ms\n", batchNo, nodeId, iteration, timProcess);
    allTimProcess += timProcess;
    onlyBatchTime += timProcess;

    if(master()) fprintf(stderr, "Batch %u fully completed on node %u took %.3lf ms (at %.3lf ms)\n", batchNo, nodeId, onlyBatchTime, allTimProcess);

    processAll(wProgram);
  }

  if(master() && printEM)
    printEngineMetrics();
}

template <typename VertexType, typename EdgeType>
void Engine<VertexType, EdgeType>::streamRun2(VertexProgram<VertexType, EdgeType>* vProgram, VertexProgram<VertexType, EdgeType>* removeApproximations, VertexProgram<VertexType, EdgeType>* exactProgram, VertexProgram<VertexType, EdgeType>* wProgram, bool smartDeletions, bool printEM) {
  assert(false); // streamRun2 must be decommissioned

  NodeManager::barrier("streamrun2");
  fprintf(stderr, "Engine::streamRun2 called.\n");

  NodeManager::startProcessing();

  unsigned numDeletions = unsigned(batchSize * (deletePercent / 100.0));
  unsigned numInsertions = batchSize - numDeletions;

  unsigned long long currentGlobalInsertCounter = 0;
  unsigned long long currentGlobalDeleteCounter = 0;

  typename std::vector<std::tuple<unsigned long long, IdType, IdType> >::iterator insIter = insertStream.begin();
  typename std::vector<std::tuple<unsigned long long, IdType, IdType> >::iterator delIter = deleteStream.begin();

  unsigned batchNo = 0;
  engineContext.setCurrentBatch(batchNo);
  quickRun(vProgram, true);
  if(master()) fprintf(stderr, "Base Batch %u completed on node %u in %u iterations took %.3lf ms\n", batchNo, nodeId, iteration, timProcess);
  allTimProcess += timProcess;

  while(batchNo++ < numBatches) {
    currentGlobalInsertCounter += numInsertions;
    currentGlobalDeleteCounter += numDeletions;

    unsigned trueAdditions = 0, trueDeletions = 0;
    shadowScheduler->clear();

    scheduler->reset();

    // Insertions
    std::set<IdType> inTopics;
    while(insIter != insertStream.end()) {
      unsigned long long ts;
      IdType from, to; 
      std::tie(ts, from, to) = *insIter;
      if(ts < currentGlobalInsertCounter) {
        addEdge(from, to, &inTopics);
        activateEndPoints(from, to, onAdd, onAddHandler);
        if(undirected) {
          addEdge(to, from, &inTopics);
          activateEndPoints(to, from, onAdd, onAddHandler);
        }
        ++insIter;
        
        if(graph.vertexPartitionIds[to] == nodeId)
          ++trueAdditions;
        continue;
      }
      break;
    }

    receiveNewGhostValues(inTopics); 

    shadowScheduler->set(scheduler->getNextBitset());
    scheduler->reset();

    // Deletions
    while(delIter != deleteStream.end()) {
      unsigned long long ts;
      IdType from, to; 
      std::tie(ts, from, to) = *delIter;
      if(ts < currentGlobalDeleteCounter) {
        if(smartDeletions) {
          LightEdge<VertexType, EdgeType> dEdge = deleteEdge2(from, to);
          activateEndPoints2(from, to, onDelete, onDeleteSmartHandler, dEdge);
          if(undirected) {
            LightEdge<VertexType, EdgeType> dEdge = deleteEdge2(to, from);
            activateEndPoints2(to, from, onDelete, onDeleteSmartHandler, dEdge);
          }
        } else {
          deleteEdge(from, to);
          activateEndPoints(from, to, onDelete, onDeleteHandler);
          if(undirected) {
            deleteEdge(to, from);
            activateEndPoints(to, from, onDelete, onDeleteHandler);
          }
        }
        ++delIter;
        
        if(graph.vertexPartitionIds[to] == nodeId)
          ++trueDeletions;
        continue;
      }
      break;
    }

    graph.compactGraph();

    trueAdditions = sillyReduce(trueAdditions, &sumReducer);
    trueDeletions = sillyReduce(trueDeletions, &sumReducer);
    if(master()) fprintf(stderr, "Batch %u constructed using %u insertions and %u deletions\n", batchNo, trueAdditions, trueDeletions);

    double onlyBatchTime = 0.0;

    //------ Phase 1 process where value and tags flow together

    IdType iTagged = scheduler->numFutureTasks();

    engineContext.setCurrentBatch(batchNo);
    //shadowScheduler->clear();
    trimScheduler->clear();

    quickRun(vProgram, true);
    if(master()) fprintf(stderr, "Batch %u phase 1 completed on node %u in %u iterations took %.3lf ms\n", batchNo, nodeId, iteration, timProcess);
    allTimProcess += timProcess;
    onlyBatchTime += timProcess;

    //------ Phase 2 process where tags are removed and values are appropriately reset

    iTagged = sillyReduce(iTagged, &sumReducer);
    IdType trimCount = sillyReduce(trimScheduler->countSetBits(), &sumReducer);
    if(master()) fprintf(stderr, "Original %u tags now spread to %u vertices; hence, wipping them off\n", iTagged, trimCount);

    //signalAll();
    scheduler->schedule(shadowScheduler);
    quickRun(removeApproximations, true);
    if(master()) fprintf(stderr, "Batch %u phase 2 completed on node %u in %u iterations took %.3lf ms\n", batchNo, nodeId, iteration, timProcess);
    allTimProcess += timProcess;
    onlyBatchTime += timProcess;

    //------ Phase 3 process where exact computations are performed

    //signalAll();
    scheduler->schedule(shadowScheduler);
    quickRun(exactProgram, true);
    if(master()) fprintf(stderr, "Batch %u phase 3 completed on node %u in %u iterations took %.3lf ms\n", batchNo, nodeId, iteration, timProcess);
    allTimProcess += timProcess;
    onlyBatchTime += timProcess;

    if(master()) fprintf(stderr, "Batch %u fully completed on node %u took %.3lf ms (at %.3lf ms)\n", batchNo, nodeId, onlyBatchTime, allTimProcess);

    //------ Writing back

    processAll(wProgram);
  }

  if(master() && printEM)
    printEngineMetrics();
}

template <typename VertexType, typename EdgeType>
void Engine<VertexType, EdgeType>::streamRun3(VertexProgram<VertexType, EdgeType>* vProgram, VertexProgram<VertexType, EdgeType>* tagProgram, VertexProgram<VertexType, EdgeType>* removeApproximations, VertexProgram<VertexType, EdgeType>* exactProgram, VertexProgram<VertexType, EdgeType>* wProgram, bool smartDeletions, bool printEM) {
  NodeManager::barrier("streamrun3");
  fprintf(stderr, "Engine::streamRun3 called.\n");

  NodeManager::startProcessing();

  unsigned numDeletions = unsigned(batchSize * (deletePercent / 100.0));
  unsigned numInsertions = batchSize - numDeletions;

  unsigned long long currentGlobalInsertCounter = 0;
  unsigned long long currentGlobalDeleteCounter = 0;

  typename std::vector<std::tuple<unsigned long long, IdType, IdType> >::iterator insIter = insertStream.begin();
  typename std::vector<std::tuple<unsigned long long, IdType, IdType> >::iterator delIter = deleteStream.begin();

  unsigned batchNo = 0;
  engineContext.setCurrentBatch(batchNo);
  quickRun(vProgram, true);
  if(master()) fprintf(stderr, "Base Batch %u completed on node %u in %u iterations took %.3lf ms\n", batchNo, nodeId, iteration, timProcess);
  allTimProcess += timProcess;

  while(batchNo++ < numBatches) {
    currentGlobalInsertCounter += numInsertions;
    currentGlobalDeleteCounter += numDeletions;

    unsigned trueAdditions = 0, trueDeletions = 0;
    shadowScheduler->clear();

    scheduler->reset();

    // Insertions
    std::set<IdType> inTopics;
    while(insIter != insertStream.end()) {
      unsigned long long ts;
      IdType from, to; 
      std::tie(ts, from, to) = *insIter;
      if(ts < currentGlobalInsertCounter) {
        addEdge(from, to, &inTopics);
        activateEndPoints(from, to, onAdd, onAddHandler);
        if(undirected) {
          addEdge(to, from, &inTopics);
          activateEndPoints(to, from, onAdd, onAddHandler);
        }
        ++insIter;
        
        if(graph.vertexPartitionIds[to] == nodeId)
          ++trueAdditions;
        continue;
      }
      break;
    }

    receiveNewGhostValues(inTopics); 

    shadowScheduler->set(scheduler->getNextBitset());
    scheduler->reset();

    // Deletions
    while(delIter != deleteStream.end()) {
      unsigned long long ts;
      IdType from, to; 
      std::tie(ts, from, to) = *delIter;
      if(ts < currentGlobalDeleteCounter) {
        if(smartDeletions) {
          LightEdge<VertexType, EdgeType> dEdge = deleteEdge2(from, to);
          activateEndPoints2(from, to, onDelete, onDeleteSmartHandler, dEdge);
          if(undirected) {
            LightEdge<VertexType, EdgeType> dEdge = deleteEdge2(to, from);
            activateEndPoints2(to, from, onDelete, onDeleteSmartHandler, dEdge);
          }
        } else {
          deleteEdge(from, to);
          activateEndPoints(from, to, onDelete, onDeleteHandler);
          if(undirected) {
            deleteEdge(to, from);
            activateEndPoints(to, from, onDelete, onDeleteHandler);
          }
        }
        ++delIter;

        if(graph.vertexPartitionIds[to] == nodeId)
          ++trueDeletions;
        continue;
      }
      break;
    }

    graph.compactGraph();

    trueAdditions = sillyReduce(trueAdditions, &sumReducer);
    trueDeletions = sillyReduce(trueDeletions, &sumReducer);
    if(master()) fprintf(stderr, "Batch %u constructed using %u insertions and %u deletions\n", batchNo, trueAdditions, trueDeletions);

    double onlyBatchTime = 0.0;

    //------ Phase 1 process where only tags flow together

    IdType iTagged = scheduler->numFutureTasks();

    engineContext.setCurrentBatch(batchNo);
    //shadowScheduler->clear();
    trimScheduler->clear();

    quickRun(tagProgram, true);
    if(master()) fprintf(stderr, "Batch %u phase 1 completed on node %u in %u iterations took %.3lf ms\n", batchNo, nodeId, iteration, timProcess);
    allTimProcess += timProcess;
    onlyBatchTime += timProcess;

    //------ Phase 2 process where tags are removed and values are appropriately reset

    iTagged = sillyReduce(iTagged, &sumReducer);
    IdType trimCount = sillyReduce(trimScheduler->countSetBits(), &sumReducer);
    if(master()) fprintf(stderr, "Original %u tags now spread to %u vertices; hence, wipping them off\n", iTagged, trimCount);

    //signalAll();
    scheduler->schedule(shadowScheduler);
    quickRun(removeApproximations, true);
    if(master()) fprintf(stderr, "Batch %u phase 2 completed on node %u in %u iterations took %.3lf ms\n", batchNo, nodeId, iteration, timProcess);
    allTimProcess += timProcess;
    onlyBatchTime += timProcess;

    //------ Phase 3 process where exact computations are performed

    //signalAll();
    scheduler->schedule(shadowScheduler);
    quickRun(exactProgram, true);
    if(master()) fprintf(stderr, "Batch %u phase 3 completed on node %u in %u iterations took %.3lf ms\n", batchNo, nodeId, iteration, timProcess);
    allTimProcess += timProcess;
    onlyBatchTime += timProcess;

    if(master()) fprintf(stderr, "Batch %u fully completed on node %u took %.3lf ms (at %.3lf ms)\n", batchNo, nodeId, onlyBatchTime, allTimProcess);

    //------ Writing back

    processAll(wProgram);
  }

  if(master() && printEM)
    printEngineMetrics();
}

template <typename VertexType, typename EdgeType>
void Engine<VertexType, EdgeType>::streamRun4(VertexProgram<VertexType, EdgeType>* vProgram, VertexProgram<VertexType, EdgeType>* trimProgram, VertexProgram<VertexType, EdgeType>* wProgram, bool smartDeletions, bool printEM) {
  NodeManager::barrier("streamrun4");
  fprintf(stderr, "Engine::streamRun4 called.\n");

  NodeManager::startProcessing();

  unsigned numDeletions = unsigned(batchSize * (deletePercent / 100.0));
  unsigned numInsertions = batchSize - numDeletions;

  unsigned long long currentGlobalInsertCounter = 0;
  unsigned long long currentGlobalDeleteCounter = 0;

  typename std::vector<std::tuple<unsigned long long, IdType, IdType> >::iterator insIter = insertStream.begin();
  typename std::vector<std::tuple<unsigned long long, IdType, IdType> >::iterator delIter = deleteStream.begin();

  unsigned batchNo = 0;
  engineContext.setCurrentBatch(batchNo);
  quickRun(vProgram, true);
  if(master()) fprintf(stderr, "Base Batch %u completed on node %u in %u iterations took %.3lf ms\n", batchNo, nodeId, iteration, timProcess);
  allTimProcess += timProcess;

  while(batchNo++ < numBatches) {
    currentGlobalInsertCounter += numInsertions;
    currentGlobalDeleteCounter += numDeletions;
    unsigned trueAdditions = 0, trueDeletions = 0;
    shadowScheduler->clear();

    scheduler->reset();

    // Insertions
    std::set<IdType> inTopics;
    while(insIter != insertStream.end()) {
      unsigned long long ts;
      IdType from, to; 
      std::tie(ts, from, to) = *insIter;
      if(ts < currentGlobalInsertCounter) {
        addEdge(from, to, &inTopics);
        activateEndPoints(from, to, onAdd, onAddHandler);
        if(undirected) {
          addEdge(to, from, &inTopics);
          activateEndPoints(to, from, onAdd, onAddHandler);
        }
        ++insIter;
        
        if(graph.vertexPartitionIds[to] == nodeId)
          ++trueAdditions;
        continue;
      }
      break;
    }

    receiveNewGhostValues(inTopics); 

    shadowScheduler->set(scheduler->getNextBitset());
    scheduler->reset();

    // Deletions
    while(delIter != deleteStream.end()) {
      unsigned long long ts;
      IdType from, to; 
      std::tie(ts, from, to) = *delIter;
      if(ts < currentGlobalDeleteCounter) {
        if(smartDeletions) {
          LightEdge<VertexType, EdgeType> dEdge = deleteEdge2(from, to);
          //if(dEdge.to.parent == dEdge.fromId) 
          activateEndPoints2(from, to, onDelete, onDeleteSmartHandler, dEdge);
          if(undirected) {
            LightEdge<VertexType, EdgeType> dEdge = deleteEdge2(to, from);
            //if(dEdge.to.parent == dEdge.fromId)
            activateEndPoints2(to, from, onDelete, onDeleteSmartHandler, dEdge);
          }
        } else {
          deleteEdge(from, to);
          activateEndPoints(from, to, onDelete, onDeleteHandler);
          if(undirected) {
            deleteEdge(to, from);
            activateEndPoints(to, from, onDelete, onDeleteHandler);
          }
        }
        ++delIter;
        
        if(graph.vertexPartitionIds[to] == nodeId)
          ++trueDeletions;
        continue;
      }
      break;
    }

    graph.compactGraph();

    trueAdditions = sillyReduce(trueAdditions, &sumReducer);
    trueDeletions = sillyReduce(trueDeletions, &sumReducer);
    if(master()) fprintf(stderr, "Batch %u constructed using %u insertions and %u deletions\n", batchNo, trueAdditions, trueDeletions);

    double onlyBatchTime = 0.0;

    //------ Phase 1 process where trimming of subtrees takes place

    IdType iTagged = scheduler->numFutureTasks();

    engineContext.setCurrentBatch(batchNo);
    //shadowScheduler->clear();
    trimScheduler->clear();

    quickRun(trimProgram, true);

    iTagged = sillyReduce(iTagged, &sumReducer);
    IdType trimCount = sillyReduce(trimScheduler->countSetBits(), &sumReducer);
    if(master()) fprintf(stderr, "Original %u tags were spread to %u vertices; hence, wipped them off\n", iTagged, trimCount);

    if(master()) fprintf(stderr, "Batch %u phase 1 completed on node %u in %u iterations took %.3lf ms\n", batchNo, nodeId, iteration, timProcess);
    allTimProcess += timProcess;
    onlyBatchTime += timProcess;

    //------ Phase 2 process where exact computations are performed

    //signalAll();
    scheduler->schedule(shadowScheduler);
    quickRun(vProgram, true);
    if(master()) fprintf(stderr, "Batch %u phase 2 completed on node %u in %u iterations took %.3lf ms\n", batchNo, nodeId, iteration, timProcess);
    allTimProcess += timProcess;
    onlyBatchTime += timProcess;

    if(master()) fprintf(stderr, "Batch %u fully completed on node %u took %.3lf ms (at %.3lf ms)\n", batchNo, nodeId, onlyBatchTime, allTimProcess);

    //------ Writing back

    processAll(wProgram);
  }

  if(master() && printEM)
    printEngineMetrics();
}

template <typename VertexType, typename EdgeType>
void Engine<VertexType, EdgeType>::streamRun5(VertexProgram<VertexType, EdgeType>* vProgram, VertexProgram<VertexType, EdgeType>* trimProgram, VertexProgram<VertexType, EdgeType>* wProgram, bool printEM) {
  NodeManager::barrier("streamrun5");
  fprintf(stderr, "Engine::streamRun5 called.\n");

  NodeManager::startProcessing();

  unsigned numDeletions = unsigned(batchSize * (deletePercent / 100.0));
  unsigned numInsertions = batchSize - numDeletions;

  unsigned long long currentGlobalInsertCounter = 0;
  unsigned long long currentGlobalDeleteCounter = 0;

  typename std::vector<std::tuple<unsigned long long, IdType, IdType> >::iterator insIter = insertStream.begin();
  typename std::vector<std::tuple<unsigned long long, IdType, IdType> >::iterator delIter = deleteStream.begin();

  unsigned batchNo = 0;
  engineContext.setCurrentBatch(batchNo);
  quickRun(vProgram, true);
  if(master()) fprintf(stderr, "Base Batch %u completed on node %u in %u iterations took %.3lf ms\n", batchNo, nodeId, iteration, timProcess);
  allTimProcess += timProcess;

  while(batchNo++ < numBatches) {
    currentGlobalInsertCounter += numInsertions;
    currentGlobalDeleteCounter += numDeletions;
    unsigned trueAdditions = 0, trueDeletions = 0;
    shadowScheduler->clear();

    scheduler->reset();

    // Insertions
    std::set<IdType> inTopics;
    while(insIter != insertStream.end()) {
      unsigned long long ts;
      IdType from, to; 
      std::tie(ts, from, to) = *insIter;
      if(ts < currentGlobalInsertCounter) {
        addEdge(from, to, &inTopics);
        activateEndPoints(from, to, onAdd, onAddHandler);
        if(undirected) {
          addEdge(to, from, &inTopics);
          activateEndPoints(to, from, onAdd, onAddHandler);
        }
        ++insIter;
        
        if(graph.vertexPartitionIds[to] == nodeId)
          ++trueAdditions;
        continue;
      }
      break;
    }

    receiveNewGhostValues(inTopics); 

    shadowScheduler->set(scheduler->getNextBitset());
    scheduler->reset(); 

    // Deletions
    while(delIter != deleteStream.end()) {
      unsigned long long ts;
      IdType from, to; 
      std::tie(ts, from, to) = *delIter;
      if(ts < currentGlobalDeleteCounter) {
        if(deleteEdge3(from, to));
        activateEndPoints(from, to, onDelete, onDeleteHandler);
        if(undirected) {
          if(deleteEdge3(to, from));
          activateEndPoints(to, from, onDelete, onDeleteHandler);
        }
        ++delIter;
        
        if(graph.vertexPartitionIds[to] == nodeId)
          ++trueDeletions;
        continue;
      }
      break;
    }

    graph.compactGraph();

    trueAdditions = sillyReduce(trueAdditions, &sumReducer);
    trueDeletions = sillyReduce(trueDeletions, &sumReducer);
    if(master()) fprintf(stderr, "Batch %u constructed using %u insertions and %u deletions\n", batchNo, trueAdditions, trueDeletions);

    double onlyBatchTime = 0.0;

    //------ Phase 1 process where trimming of subtrees takes place

    IdType iTagged = scheduler->numFutureTasks();

    engineContext.setCurrentBatch(batchNo);
    //shadowScheduler->clear();
    trimScheduler->clear();

    //timProcess = 0.0;
    //if(false)
    quickRun(trimProgram, true);

    iTagged = sillyReduce(iTagged, &sumReducer);
    IdType trimCount = sillyReduce(trimScheduler->countSetBits(), &sumReducer);
    if(master()) fprintf(stderr, "Original %u tags were spread to %u vertices; hence, wipped them off\n", iTagged, trimCount);

    if(master()) fprintf(stderr, "Batch %u phase 1 completed on node %u in %u iterations took %.3lf ms\n", batchNo, nodeId, iteration, timProcess);
    allTimProcess += timProcess;
    onlyBatchTime += timProcess;

    //------ Phase 2 process where exact computations are performed

    //signalAll();
    scheduler->schedule(shadowScheduler);
    quickRun(vProgram, true);
    if(master()) fprintf(stderr, "Batch %u phase 2 completed on node %u in %u iterations took %.3lf ms\n", batchNo, nodeId, iteration, timProcess);
    allTimProcess += timProcess;
    onlyBatchTime += timProcess;

    if(master()) fprintf(stderr, "Batch %u fully completed on node %u took %.3lf ms (at %.3lf ms)\n", batchNo, nodeId, onlyBatchTime, allTimProcess);

    //------ Writing back

    processAll(wProgram);
  }

  if(master() && printEM)
    printEngineMetrics();
}

template <typename VertexType, typename EdgeType>
void Engine<VertexType, EdgeType>::processAll(VertexProgram<VertexType, EdgeType>* vProgram) {
  vProgram->beforeIteration(engineContext);

  for(IdType i=0; i<graph.numLocalVertices; ++i)
    vProgram->processVertex(graph.vertices[i]);

  vProgram->afterIteration(engineContext);
}

template <typename VertexType, typename EdgeType>
void Engine<VertexType, EdgeType>::worker(unsigned tid, void* args) {
  static bool compHalt = false;
  static bool pushWait = false;
  compHalt = false;
  pushWait = false;
  //engineContext.setTooLong(false);
  while(1) {      // Outer while loop
    IdType vid; bool found = false;
    lockCurrId.lock();
    while(1) {      // Inner while loop
      found = false;
      if(currId >= graph.numLocalVertices) {
        if(tid != 0) {
          lockCurrId.unlock();

          barComp.wait();                   // This is "A". Let everyone reach to this point, then only master will work
          barComp.wait();                   // Master has finished its work

          if(compHalt)
            break;

          lockCurrId.lock();
          continue;
        } else {    // tid == 0
          lockCurrId.unlock();

          barComp.wait();                   // This is "A"

          if(firstIteration == false) {
            ++iteration;

            if(iteration % poFrequency == 0) {
              remPushOut = numNodes;
              VertexType stub = VertexType(2, 0);
              CommManager::dataPushOut(PUSHOUT_REQ_ME, stub.data(), sizeof(FeatType) * stub.size());

              pushWait = true; 
            }
          }
          firstIteration = false;

          fprintf(stderr, "Starting iteration %u at %.3lf ms\n", iteration, timProcess + getTimer()); 
          //if(iteration > 1000)
          //engineContext.setTooLong(true);

          bool hltBool = ((timProcess + getTimer() > 500) && !(scheduler->anyScheduledTasks()));  // after 1 sec only!

          if(!hltBool) {
            scheduler->newIteration();

            currId = 0;     // This is unprotected by lockCurrId because only master executes this code while workers are on barriers
            lockCurrId.lock();
            barComp.wait();
          } else { // deciding to halt
            fprintf(stderr, "Deciding to halt in this iteration (%u)\n", iteration);
            NodeManager::barrier(COMM_BARRIER);

            double bCompTime = -getTimer();
            pthread_mutex_lock(&mtxDataWaiter);
            compDone = true;
            pthread_mutex_unlock(&mtxDataWaiter);

            fprintf(stderr, "1. Comp waiting on barCompData\n");
            barCompData.wait();                   // Wake data up

            fprintf(stderr, "2. Comp waiting on barCompData after %.3lf ms barCompData time\n", bCompTime + getTimer());
            barCompData.wait();                   // Wait for data's decision

            pthread_mutex_lock(&mtxDataWaiter);
            compDone = false;
            compHalt = halt;
            die = halt;
            pthread_mutex_unlock(&mtxDataWaiter);
            fprintf(stderr, "3. Comp waiting on barCompData after %.3lf ms barCompData time\n", bCompTime + getTimer());
            barCompData.wait();                   // Wake data up

            fprintf(stderr, "4. Comp waiting on barCompData after %.3lf ms barCompData time\n", bCompTime + getTimer());
            barCompData.wait();                   // Allow only data to proceed further before this barrier so that it can aquire the lock

            fprintf(stderr, "barCompData time = %.3lf ms\n", bCompTime + getTimer());

            if(compHalt) {
              barComp.wait();
              break;
            }

            if(scheduler->anyScheduledTasks()) {     // TODO: does this assert hold true?
              scheduler->newIteration();
              currId = 0; // This is unprotected by lockCurrId because only master executes this code while workers are on barriers
            }

            lockCurrId.lock();
            barComp.wait();
            continue;

          } // end of else: deciding to halt
        } // end of else: tid == 0
      } // end of if: currId >= graph.numLocalVertices

      if(pushWait) {  // Note tid is always 0 here because it is the one who will have lockCurrId held when pushWait = true
        assert(tid == 0);
        while(remPushOut > 0) { }; // spinwait? omg?
        assert(remPushOut == 0);
        pushWait = false;
      }

      if(scheduler->isScheduled(currId)) {
        vid = currId;
        found = true;
        ++currId;
        break;
      }
      ++currId;
    }   // Inner while loop
    lockCurrId.unlock();

    if(!found) {
      fprintf(stderr, "Nothing to work on. Leaving.\n");
      break;
    }

    Vertex<VertexType, EdgeType>& v = graph.vertices[vid];

    assert(scheduler->isScheduled(vid));
    bool ret = firstIteration | vertexProgram->update(v, engineContext);   // this is a very important change. firstIteration ensures that tags flow out.
    if(ret) {
      bool remoteScat = true;
      for(unsigned i=0; i<v.numOutEdges(); ++i) {
        if(v.getOutEdge(i).getEdgeLocation() == LOCAL_EDGE_TYPE)
          scheduler->schedule(v.getOutEdge(i).destId());
        else {
          if(remoteScat) {
            CommManager::dataPushOut(graph.localToGlobalId[vid], (void*)v.vertexData.data(), sizeof(FeatType) * v.vertexData.size());
            remoteScat = false;
          }
        }
      }
    }
  } // Outer while loop
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

template <typename VertexType, typename EdgeType>
void Engine<VertexType, EdgeType>::dataCommunicator(unsigned tid, void* args) {
  IdType vid;
  VertexType value;

  while(1) {
    if(!CommManager::dataPullIn(vid, value)) {
      if(!compDone)
        continue;

      double dCommTime = -getTimer();
      NodeManager::barrier(DATACOMM_BARRIER);
      fprintf(stderr, "DATACOMM_BARRIER time = %.3lf ms\n", dCommTime + getTimer());
      fprintf(stderr, "1. Data waiting on barCompData\n");
      barCompData.wait();
      pthread_mutex_lock(&mtxDataWaiter);

      assert(compDone);
      //fprintf(stderr, "compDone is true\n");

      CommManager::dataPushOut(ITHINKIAMDONE, (void*)value.data(), value.size() * sizeof(FeatType));

      //1. Send out i from your data channel
      //2. The idea is to receive n-1 ack-i, but in between you can receive other machine's j so you simply send ack-j
      unsigned rem = numNodes;
      bool thinkHalt = !(scheduler->anyScheduledTasks());
      halt = true;
      while(rem > 0) {
        IdType mType;
        while(!CommManager::dataPullIn(mType, value)) { } // spinwait? omg?

        if(mType == ITHINKIAMDONE) {
          --rem;
        } else if(mType == IAMNOTDONE) {
          assert(false);
          thinkHalt = false;
        } 
        else if(mType == IAMDONE) {
          assert(false);
        } else if((mType >= PUSHOUT_REQ_BEGIN) && (mType < PUSHOUT_REQ_END)) {
          IdType response = mType - PUSHOUT_REQ_BEGIN + PUSHOUT_RESP_BEGIN;
          CommManager::dataPushOut(response, (void*)value.data(), value.size() * sizeof(FeatType));
        } else if((mType >= PUSHOUT_RESP_BEGIN) && (mType < PUSHOUT_RESP_END)) {
          if(mType == PUSHOUT_RESP_ME) {
            --remPushOut;
          }
        } else {
          vid = mType;
          conditionalUpdateGhostVertex(vid, value);
          thinkHalt = false;
        }
      }

      NodeManager::barrier(DATACOMM_BARRIER);     // This is required.

      if(thinkHalt) {
        CommManager::dataPushOut(IAMDONE, (void*)value.data(), value.size() * sizeof(FeatType));
      } else {
        CommManager::dataPushOut(IAMNOTDONE, (void*)value.data(), value.size() * sizeof(FeatType));
        halt = false;
      }

      IdType mType; unsigned totalRem = numNodes; unsigned finalRem = numNodes;
      while(totalRem > 0) {
        while(!CommManager::dataPullIn(mType, value)) { } // spinlock? omg?

        if(mType == IAMDONE) {
          --finalRem;
          --totalRem;
        }
        else if(mType == IAMNOTDONE) {
          halt = false;
          --totalRem;
        } else if(mType == ITHINKIAMDONE) {    // Cannot be AREYOUDONE, ITHINKIAMDONE, vid
          fprintf(stderr, "Something is fishy: mType = %u\n", mType);
          assert(false);
        } else if((mType >= PUSHOUT_REQ_BEGIN) && (mType < PUSHOUT_REQ_END)) {
          halt = false;
          IdType response = mType - PUSHOUT_REQ_BEGIN + PUSHOUT_RESP_BEGIN;
          CommManager::dataPushOut(response, (void*)value.data(), value.size() * sizeof(FeatType));
        } else if((mType >= PUSHOUT_RESP_BEGIN) && (mType < PUSHOUT_RESP_END)) {
          if(mType == PUSHOUT_RESP_ME) {
            --remPushOut;
          }
        } else {
          assert(false);
          vid = mType;

          conditionalUpdateGhostVertex(vid, value);
          halt = false;
        }
      }

      NodeManager::barrier(DATACOMM_BARRIER);     // This is required because one of the machines can go so far ahead that we may start receiving data from worker threads on that machine and hit asserts above (because we are looking for IAMDONE/IAMNOTDONE only)

      if(finalRem != 0)
        assert(halt == false);

      pthread_mutex_unlock(&mtxDataWaiter);
      fprintf(stderr, "2. Data waiting on barCompData\n");
      barCompData.wait();                       // Wake up comp to see my decision

      fprintf(stderr, "3. Data waiting on barCompData\n");
      barCompData.wait();                       // Wait for them to say compDone = false

      pthread_mutex_lock(&mtxDataWaiter);
      assert(compDone == false);
      pthread_mutex_unlock(&mtxDataWaiter);
      fprintf(stderr, "4. Data waiting on barCompData\n");
      barCompData.wait();                       // I acquired the lock and saw compDone = false;

      fprintf(stderr, "4.5 Data left barCompData\n");

      if(halt) {
        fprintf(stderr, "dataCommunicator halt is true, hence leaving\n");
        break;
      }

      continue;
    }

    //4. Check for message type. if nak, ignore, if j send out nak-j, if regular, do below

    if(vid == IAMDONE || vid == IAMNOTDONE || vid == ITHINKIAMDONE) {
      assert(false);
    }

    if((vid >= PUSHOUT_REQ_BEGIN) && (vid < PUSHOUT_REQ_END)) {
      IdType response = vid - PUSHOUT_REQ_BEGIN + PUSHOUT_RESP_BEGIN;
      CommManager::dataPushOut(response, value.data(), value.size() * sizeof(FeatType)); 
      continue;
    }

    if((vid >= PUSHOUT_RESP_BEGIN) && (vid < PUSHOUT_RESP_END)) {
      if(vid == PUSHOUT_RESP_ME) {
        --remPushOut;
      }
      continue;
    }

    conditionalUpdateGhostVertex(vid, value);
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
