#ifndef __ENGINE_CPP__
#define __ENGINE_CPP__


#include <iostream>
#include <fstream>
#include <cmath>
#include <boost/program_options.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/algorithm/string/classification.hpp>    // Include boost::for is_any_of.
#include <boost/algorithm/string/split.hpp>
#include <string>
#include <cstdlib>
#include <omp.h>
#include "engine.hpp"


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
VertexProgram<VertexType, EdgeType>* Engine<VertexType, EdgeType>::vertexProgram = NULL;

template <typename VertexType, typename EdgeType>
EdgeType (*Engine<VertexType, EdgeType>::edgeWeight) (IdType, IdType) = NULL;

template <typename VertexType, typename EdgeType>
IdType Engine<VertexType, EdgeType>::currId = 0;

template <typename VertexType, typename EdgeType>
Lock Engine<VertexType, EdgeType>::lockCurrId;

template <typename VertexType, typename EdgeType>
Lock Engine<VertexType, EdgeType>::lockRecvWaiters;

template <typename VertexType, typename EdgeType>
Cond Engine<VertexType, EdgeType>::condRecvWaitersEmpty;

template <typename VertexType, typename EdgeType>
Lock Engine<VertexType, EdgeType>::lockHalt;

template <typename VertexType, typename EdgeType>
unsigned Engine<VertexType, EdgeType>::nodeId;

template <typename VertexType, typename EdgeType>
unsigned Engine<VertexType, EdgeType>::numNodes;

template <typename VertexType, typename EdgeType>
std::map<IdType, unsigned> Engine<VertexType, EdgeType>::recvWaiters;

template <typename VertexType, typename EdgeType>
Barrier Engine<VertexType, EdgeType>::barComp;

template <typename VertexType, typename EdgeType>
VertexType Engine<VertexType, EdgeType>::defaultVertex;

template <typename VertexType, typename EdgeType>
EdgeType Engine<VertexType, EdgeType>::defaultEdge;

template <typename VertexType, typename EdgeType>
unsigned Engine<VertexType, EdgeType>::iteration = 0;

template <typename VertexType, typename EdgeType>
bool Engine<VertexType, EdgeType>::undirected = false;

template <typename VertexType, typename EdgeType>
bool Engine<VertexType, EdgeType>::halt = false;

template <typename VertexType, typename EdgeType>
double Engine<VertexType, EdgeType>::timeProcess = 0.0;

template <typename VertexType, typename EdgeType>
double Engine<VertexType, EdgeType>::allTimeProcess = 0.0;

template <typename VertexType, typename EdgeType>
double Engine<VertexType, EdgeType>::timeInit = 0.0;


/**
 *
 * Initialize the engine with the given command line arguments.
 * 
 */
template <typename VertexType, typename EdgeType>
void
Engine<VertexType, EdgeType>::init(int argc, char *argv[], VertexType dVertex, EdgeType dEdge, EdgeType (*eWeight) (IdType, IdType)) {
    printLog(nodeId, "Engine starts initialization...");
    timeInit = -getTimer();

    // Parse the command line arguments.
    parseArgs(argc, argv);
    
    // Initialize the node manager and communication manager.
    NodeManager::init(ZKHOST_FILE, HOST_FILE);
    CommManager::init();
    nodeId = NodeManager::getNodeId();
    numNodes = NodeManager::getNumNodes();
    assert(numNodes <= 256);    // Cluster size limitation.

    defaultVertex = dVertex;
    defaultEdge = dEdge;
    edgeWeight = eWeight;

    // Read in the graph and subscribe vertex global ID topics.
    std::set<IdType> inTopics;
    std::vector<IdType> outTopics;
    readGraphBS(graphFile, inTopics, outTopics);
    graph.printGraphMetrics();

    // Read in initial features from the features file.
    readFeaturesFile(featuresFile);

    // Initialize synchronization utilities.
    lockCurrId.init();
    lockRecvWaiters.init();
    condRecvWaitersEmpty.init(lockRecvWaiters);
    lockHalt.init();

    // Initialize computation thread barrier.
    barComp.init(cThreads);

    // Create computation workers thread pool.
    computePool = new ThreadPool(cThreads);
    computePool->createPool();

    CommManager::flushControl();
    CommManager::flushData();

    // Create data communicators thread pool.
    dataPool = new ThreadPool(dThreads);
    dataPool->createPool();

    // Compact the graph.
    graph.compactGraph();

    timeInit += getTimer();
    printLog(nodeId, "Engine initialization complete.");

    // Make sure all nodes finish initailization.
    NodeManager::barrier(INIT_BARRIER);
}



/**
 *
 * Whether I am the master mode or not.
 * 
 */
template <typename VertexType, typename EdgeType>
bool
Engine<VertexType, EdgeType>::master() {
    return NodeManager::amIMaster();
}


/**
 *
 * Run the engine with the given vertex program (whose `update()` member function is customized).
 * Will start a bunch of worker threads and a bunch of data communicator threads.
 * 
 */
template <typename VertexType, typename EdgeType>
void
Engine<VertexType, EdgeType>::run(VertexProgram<VertexType, EdgeType> *vProgram, bool printEM) {
    
    // Make sure engines on all machines start running.
    NodeManager::barrier(RUN_BARRIER); 
    printLog(nodeId, "Engine starts running...\n");

    timeProcess = -getTimer();

    // Change phase to processing.
    NodeManager::startProcessing();

    // Set initial conditions.
    vertexProgram = vProgram;
    currId = 0;
    iteration = 0;
    halt = false;

    // Start data communicators.
    dataPool->perform(dataCommunicator);

    // Start workers.
    computePool->perform(worker);

    // Join all workers.
    computePool->sync();

    timeProcess += getTimer();

    // Join all data communicators.
    dataPool->sync();

    printLog(nodeId, "Engine completes the processing at iteration %u.\n", iteration);
    if (master() && printEM)
        printEngineMetrics();
}


/**
 *
 * Process all vertices using the given vertex program. Useful for a writer program.
 * 
 */
template <typename VertexType, typename EdgeType>
void
Engine<VertexType, EdgeType>::processAll(VertexProgram<VertexType, EdgeType> *vProgram) {
    vProgram->beforeIteration(iteration);

    // Loop through all local vertices and process it.
    for (IdType i = 0; i < graph.numLocalVertices; ++i)
        vProgram->processVertex(graph.vertices[i]);

    vProgram->afterIteration(iteration);
}


/**
 *
 * Destroy the engine.
 * 
 */
template <typename VertexType, typename EdgeType>
void
Engine<VertexType, EdgeType>::destroy() {
    NodeManager::destroy();
    CommManager::destroy();
    computePool->destroyPool();
    dataPool->destroyPool();

    lockCurrId.destroy();
    lockRecvWaiters.destroy();
    condRecvWaitersEmpty.destroy();
    lockHalt.destroy();
}


/////////////////////////////////////////////////
// Below are private functions for the engine. //
/////////////////////////////////////////////////


/**
 *
 * Major part of the engine's computation logic is done by workers. When the engine runs it wakes threads up from the thread pool
 * and assign a worker function for each.
 * 
 */
template <typename VertexType, typename EdgeType>
void
Engine<VertexType, EdgeType>::worker(unsigned tid, void *args) {

    // Outer while loop. Looping infinitely, looking for a new task to handle.
    while (1) {

        // Get current vertex that need to be handled.
        lockCurrId.lock();
        IdType local_vid = currId++;
        lockCurrId.unlock();

        // All local vertices have been processed. Hit the barrier and wait for next iteration / decide to halt.
        if (local_vid >= graph.numLocalVertices) {

            // Non-master threads.
            if (tid != 0) {

                //## Worker barrier 1: Everyone reach to this point, then only master will work. ##//
                barComp.wait();

                //## Worker barrier 2: Master has finished its checking work. ##//
                barComp.wait();

                // If master says halt then go to death; else continue for the new iteration.
                if (halt)
                    return;
                else
                    continue;

            // Master thread (tid == 0).
            } else {

                //## Worker barrier 1: Everyone reach to this point, then only master will work. ##//
                barComp.wait();

                // Wait for all remote schedulings sent by me to be handled.
                lockRecvWaiters.lock();
                if (!recvWaiters.empty())
                    condRecvWaitersEmpty.wait();
                lockRecvWaiters.unlock();

                //## Global Iteration barrier. ##//
                NodeManager::barrier(LAYER_BARRIER);

                // Yes there are further scheduled vertices. Please start a new iteration.
                if (iteration + 1 < NUM_LAYERS) {
                    printLog(nodeId, "Starting a new iteration %u at %.3lf ms...\n", iteration, timeProcess + getTimer());

                    // Step forward to a new iteration. 
                    ++iteration;

                    // Reset current id.
                    currId = 0;       // This is unprotected by lockCurrId because only master executes.

                    //## Worker barrier 2: Starting a new iteration. ##//
                    barComp.wait();

                    continue;

                // No more, so deciding to halt. But still needs the communicator to check if there will be further scheduling invoked by ghost
                // vertices. If so we are stilling going to the next iteration.
                } else {
                    printLog(nodeId, "Deciding to halt at iteration %u...\n", iteration);

                    // Set this to signal data communicator to end its life.
                    lockHalt.lock();
                    halt = true;
                    lockHalt.unlock();

                    //## Worker barrier 2: Going to die. ##//
                    barComp.wait();

                    return;
                }
            }
        }

        // Doing the task.
        Vertex<VertexType, EdgeType>& v = graph.vertices[local_vid];
        vertexProgram->update(v, iteration);

        // If there are any remote edges, should send this vid to others for their ghost's update.
        for (unsigned i = 0; i < v.numOutEdges(); ++i) {
            if (v.getOutEdge(i).getEdgeLocation() == REMOTE_EDGE_TYPE) {
                IdType global_vid = graph.localToGlobalId[local_vid];

                // Record that this vid gets broadcast in this iteration. Should wait for its corresponding respond.
                lockRecvWaiters.lock();
                recvWaiters[global_vid] = numNodes;
                lockRecvWaiters.unlock();

                CommManager::dataPushOut(global_vid, (void *) v.data().data(), sizeof(FeatType) * v.data().size());
                break;
            }
        }
    }
}


/**
 *
 * Major part of the engine's communication logic is done by data threads. These threads loop asynchronously with computation workers.
 * 
 */
template <typename VertexType, typename EdgeType>
void
Engine<VertexType, EdgeType>::dataCommunicator(unsigned tid, void *args) {
    IdType topic;
    VertexType value;

    // While loop, looping infinitely to get the next message.
    while(1) {

        // No message in queue.
        if (!CommManager::dataPullIn(topic, value)) {

            // Computation workers done their work, so communicator goes to death as well.
            if (halt)
                return;

        // Pull in the next message, and process this message.
        } else {
            IdType global_vid = topic;

            // A normal ghost value broadcast.
            if (value.size() != 1) {

                // Update the ghost vertex if it is one of mine.
                if (graph.ghostVertices.find(global_vid) != graph.ghostVertices.end())
                    graph.updateGhostVertex(global_vid, value);

                // TODO: Using 1-D vec to indicate a respond here. Needs change.
                VertexType recv_stub = VertexType(1, 0);
                CommManager::dataPushOut(global_vid, (void *) recv_stub.data(), sizeof(FeatType) * recv_stub.size());

            // A respond to a broadcast, and the topic vertex is in my local vertices. I should update the
            // corresponding recvWaiter's value. If waiters become empty, send a signal in case the workers are
            // waiting on it to be empty at the iteration barrier.
            } else if (graph.globalToLocalId.find(topic) != graph.globalToLocalId.end()) {
                lockRecvWaiters.lock();
                assert(recvWaiters.find(global_vid) != recvWaiters.end());
                --recvWaiters[global_vid];
                if (recvWaiters[global_vid] == 0) {
                    recvWaiters.erase(global_vid);
                    if (recvWaiters.empty())
                        condRecvWaitersEmpty.signal();
                }
                lockRecvWaiters.unlock();
            }
        }
    }
}


/**
 *
 * Print engine metrics of processing time.
 * 
 */
template <typename VertexType, typename EdgeType>
void
Engine<VertexType, EdgeType>::printEngineMetrics() {
    printLog(nodeId, "Engine Metrics: Init time = %.3lf ms\n", timeInit);
    printLog(nodeId, "Engine Metrics: Processing time = %.3lf ms\n", allTimeProcess);
}


/**
 *
 * Print my graph's metrics.
 * 
 */
template <typename VertexType, typename EdgeType>
void
Engine<VertexType, EdgeType>::printGraphMetrics() {
    printLog(nodeId, "Graph Metrics: numGlobalVertices = %u\n", graph.numGlobalVertices);
    printLog(nodeId, "Graph Metrics: numGlobalEdges = %llu\n", graph.numGlobalEdges);
    printLog(nodeId, "Graph Metrics: numLocalVertices = %u\n", graph.numLocalVertices);
}


/**
 *
 * Parse command line arguments.
 * 
 */
template <typename VertexType, typename EdgeType>
void
Engine<VertexType, EdgeType>::parseArgs(int argc, char *argv[]) {
    boost::program_options::options_description desc("Allowed options");
    desc.add_options()
        ("help", "Produce help message")

        ("config", boost::program_options::value<std::string>()->default_value(std::string(DEFAULT_CONFIG_FILE), DEFAULT_CONFIG_FILE), "Config file")

        ("graphfile", boost::program_options::value<std::string>(), "Graph file")
        ("featuresfile", boost::program_options::value<std::string>(), "Features file")

        ("undirected", boost::program_options::value<unsigned>()->default_value(unsigned(ZERO), ZERO_STR), "Graph type")

        ("dthreads", boost::program_options::value<unsigned>()->default_value(unsigned(NUM_DATA_THREADS), NUM_DATA_THREADS_STR), "Number of data threads")
        ("cthreads", boost::program_options::value<unsigned>()->default_value(unsigned(NUM_COMP_THREADS), NUM_COMP_THREADS_STR), "Number of compute threads")
        ("dport", boost::program_options::value<unsigned>()->default_value(unsigned(DATA_PORT), DATA_PORT_STR), "Port for data communication")
        ("cport", boost::program_options::value<unsigned>()->default_value(unsigned(CONTROL_PORT_START), CONTROL_PORT_START_STR), "Port start for control communication")
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
    dThreads = vm["dthreads"].as<unsigned>();   // Communicator threads.

    assert(vm.count("cthreads"));
    cThreads = vm["cthreads"].as<unsigned>();   // Computation threads.

    assert(vm.count("graphfile"));
    graphFile = vm["graphfile"].as<std::string>();

    assert(vm.count("featuresfile"));
    featuresFile = vm["featuresfile"].as<std::string>();

    assert(vm.count("undirected"));
    undirected = (vm["undirected"].as<unsigned>() == 0) ? false : true;

    assert(vm.count("dport"));
    unsigned data_port = vm["dport"].as<unsigned>();
    CommManager::setDataPort(data_port);

    assert(vm.count("cport"));
    unsigned control_port = vm["cport"].as<unsigned>();
    CommManager::setControlPortStart(control_port);

    printLog(nodeId, "Printing parsed configuration:");
    printLog(nodeId, "  config = %s\n", cFile.c_str());
    printLog(nodeId, "  dThreads = %u\n", dThreads);
    printLog(nodeId, "  cThreads = %u\n", cThreads);
    printLog(nodeId, "  graphFile = %s\n", graphFile.c_str());
    printLog(nodeId, "  featuresFile = %s\n", featuresFile.c_str());
    printLog(nodeId, "  undirected = %s\n", undirected ? "true" : "false");
    printLog(nodeId, "  data port set -> %u", data_port);
    printLog(nodeId, "  control port set -> %u", control_port);
}


/**
 *
 * Read in the initial features file.
 * 
 */
template <typename VertexType, typename EdgeType>
void
Engine<VertexType, EdgeType>::readFeaturesFile(std::string& featuresFileName) {
    std::ifstream infile(featuresFileName.c_str());
    if(!infile.good())
        printLog(nodeId, "Cannot open feature file: %s\n", featuresFileName.c_str());

    assert(infile.good());

    static const std::size_t LINE_BUFFER_SIZE=8192;
    char buffer[LINE_BUFFER_SIZE];

    std::vector<VertexType> feature_mat;

    // Loop through each line.
    while (infile.eof()!=true){
        infile.getline(buffer,LINE_BUFFER_SIZE);
        std::vector<std::string> splited_strings;
        VertexType feature_vec=VertexType();
        std::string line(buffer);

        // Split each line into numbers.
        boost::split(splited_strings, line, boost::is_any_of(", "), boost::token_compress_on);

        for (auto it = splited_strings.begin(); it != splited_strings.end(); it++) {
            if (it->data()[0] != 0) // Check null char at the end.
                feature_vec.push_back(std::stof(it->data()));
        }
        feature_mat.push_back(feature_vec);
    }

    // Set the vertices' initial values.
    for (std::size_t i = 0; i < feature_mat.size(); ++i){
        
        // Is ghost node.
        auto git = graph.ghostVertices.find(i);
        if (git != graph.ghostVertices.end()){
            graph.ghostVertices[i].setData(feature_mat[i]);
            continue;
        }
        
        // Is local node.
        auto lit = graph.globalToLocalId.find(i);
        if (lit != graph.globalToLocalId.end()){
            graph.vertices[lit->second].setData(feature_mat[i]);
            continue;
        }
    }
}


/**
 *
 * Read in the partition file.
 * 
 */
template <typename VertexType, typename EdgeType>
void
Engine<VertexType, EdgeType>::readPartsFile(std::string& partsFileName, Graph<VertexType, EdgeType>& lGraph) {
    std::ifstream infile(partsFileName.c_str());
    if(!infile.good())
        printLog(nodeId, "Cannot open patition file: %s\n", partsFileName.c_str());

    assert(infile.good());

    short partId;
    IdType lvid = 0;
    IdType gvid = 0;

    std::string line;
    while (std::getline(infile, line)) {
        if (line.size() == 0 || (line[0] < '0' || line[0] > '9'))
            continue;

        std::istringstream iss(line);
        if (!(iss >> partId))
            break;

        lGraph.vertexPartitionIds.push_back(partId);

        if (partId == nodeId) {
            lGraph.localToGlobalId[lvid] = gvid;
            lGraph.globalToLocalId[gvid] = lvid;

            ++lvid;
        }
        ++gvid;
    }

    lGraph.numGlobalVertices = gvid;
    lGraph.numLocalVertices = lvid;
}


/**
 *
 * Process an edge read from the binary snap file.
 * 
 */
template <typename VertexType, typename EdgeType>
void
Engine<VertexType, EdgeType>::processEdge(IdType& from, IdType& to, Graph<VertexType, EdgeType>& lGraph, std::set<IdType> *inTopics, std::set<IdType> *oTopics) {
    if (lGraph.vertexPartitionIds[from] == nodeId) {
        IdType lFromId = lGraph.globalToLocalId[from];
        IdType toId;
        EdgeLocationType eLocation;

        if (lGraph.vertexPartitionIds[to] == nodeId) {
            toId = lGraph.globalToLocalId[to];
            eLocation = LOCAL_EDGE_TYPE;
        } else {
            toId = to;
            eLocation = REMOTE_EDGE_TYPE;
            lGraph.vertices[lFromId].vertexLocation = BOUNDARY_VERTEX;

            if (oTopics != NULL)
                oTopics->insert(from);
        }

        if(edgeWeight != NULL)
            lGraph.vertices[lFromId].outEdges.push_back(OutEdge<EdgeType>(toId, eLocation, edgeWeight(from, to)));
        else
            lGraph.vertices[lFromId].outEdges.push_back(OutEdge<EdgeType>(toId, eLocation, defaultEdge));
    }

    if (lGraph.vertexPartitionIds[to] == nodeId) {
        IdType lToId = lGraph.globalToLocalId[to];
        IdType fromId;
        EdgeLocationType eLocation;

        if (lGraph.vertexPartitionIds[from] == nodeId) {
            fromId = lGraph.globalToLocalId[from];
            eLocation = LOCAL_EDGE_TYPE;
        } else {
            fromId = from;
            eLocation = REMOTE_EDGE_TYPE;

            typename std::map< IdType, GhostVertex<VertexType> >::iterator gvit = lGraph.ghostVertices.find(from);
            if (gvit == lGraph.ghostVertices.end()) {
                lGraph.ghostVertices[from] = GhostVertex<VertexType>(defaultVertex);
                gvit = lGraph.ghostVertices.find(from);
            }
            gvit->second.outEdges.push_back(lToId);

            if (inTopics != NULL)
                inTopics->insert(from);
        }

        if (edgeWeight != NULL)
            lGraph.vertices[lToId].inEdges.push_back(InEdge<EdgeType>(fromId, eLocation, edgeWeight(from, to)));
        else
            lGraph.vertices[lToId].inEdges.push_back(InEdge<EdgeType>(fromId, eLocation, defaultEdge));
    }
}


/**
 *
 * Set the normalization factors on all edges.
 * 
 */
template<typename VertexType, typename EdgeType>
void
Engine<VertexType, EdgeType>::setEdgeNormalizations() {
    for (Vertex<VertexType, EdgeType>& vertex : graph.vertices) {
        unsigned dstDeg = vertex.numInEdges() + 1;
        float dstNorm = std::pow(dstDeg, -.5);
        for (InEdge<EdgeType>& e : vertex.inEdges) {
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


/**
 *
 * Finds the in degree of all ghost vertices.
 * 
 */
template <typename VertexType, typename EdgeType>
void
Engine<VertexType, EdgeType>::findGhostDegrees(std::string& fileName) {
    std::ifstream infile(fileName.c_str(), std::ios::binary);
    if (!infile.good())
        printLog(nodeId, "Cannot open BinarySnap file: %s\n", fileName.c_str());

    assert(infile.good());

    BSHeaderType<IdType> bsHeader;
    infile.read((char *) &bsHeader, sizeof(bsHeader));

    IdType srcdst[2];
    while (infile.read((char *) srcdst, bsHeader.sizeOfVertexType * 2)) {
        if (srcdst[0] == srcdst[1])
            continue;

        typename std::map< IdType, GhostVertex<VertexType> >::iterator gvit = graph.ghostVertices.find(srcdst[1]);
        if (gvit != graph.ghostVertices.end())
            (gvit->second).incrementDegree();
    }
    
    infile.close();
}


/**
 *
 * Read and parse the graph from the graph binary snap file.
 * 
 */
template <typename VertexType, typename EdgeType>
void
Engine<VertexType, EdgeType>::readGraphBS(std::string& fileName, std::set<IdType>& inTopics, std::vector<IdType>& outTopics) {
    
    // Read in the partition file.
    std::string partsFileName = fileName + PARTS_EXT;
    readPartsFile(partsFileName, graph);

    // Initialize the graph based on the partition info.
    graph.vertices.resize(graph.numLocalVertices);
    for (IdType i = 0; i < graph.numLocalVertices; ++i) {
        graph.vertices[i].localIdx = i;
        graph.vertices[i].globalIdx = graph.localToGlobalId[i];
        graph.vertices[i].vertexLocation = INTERNAL_VERTEX;
        graph.vertices[i].vertexData.clear();
        graph.vertices[i].vertexData.push_back(defaultVertex);
        graph.vertices[i].graph = &graph;
    }

    // Read in the binary snap edge file.
    std::string edgeFileName = fileName + EDGES_EXT;
    std::ifstream infile(edgeFileName.c_str(), std::ios::binary);
    if(!infile.good())
        printLog(nodeId, "Cannot open BinarySnap file: %s\n", edgeFileName.c_str());

    assert(infile.good());

    BSHeaderType<IdType> bSHeader;
    infile.read((char *) &bSHeader, sizeof(bSHeader));
    assert(bSHeader.sizeOfVertexType == sizeof(IdType));

    // Loop through all edges and process them.
    std::set<IdType> oTopics;
    graph.numGlobalEdges = 0;
    IdType srcdst[2];
    while (infile.read((char *) srcdst, bSHeader.sizeOfVertexType * 2)) {
        if (srcdst[0] == srcdst[1])
            continue;

        processEdge(srcdst[0], srcdst[1], graph, &inTopics, &oTopics);
        if (undirected)
            processEdge(srcdst[1], srcdst[0], graph, &inTopics, &oTopics);
        ++graph.numGlobalEdges;
    }

    infile.close();

    // Extra works added.
    findGhostDegrees(edgeFileName);
    setEdgeNormalizations();

    typename std::set<IdType>::iterator it;
    for (it = oTopics.begin(); it != oTopics.end(); ++it)
        outTopics.push_back(*it);
}


#endif // __ENGINE_CPP__
