#include <iostream>
#include <fstream>
#include <cmath>
#include <boost/program_options.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/algorithm/string/classification.hpp>    // Include boost::for is_any_of.
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/trim.hpp>
#include <string>
#include <thread>
#include <cstdlib>
#include <omp.h>
#include <cerrno>
#include "engine.hpp"


/** Extern class-wide fields. */
Graph Engine::graph;
ThreadPool* Engine::dataPool = NULL;
unsigned Engine::dThreads = NUM_DATA_THREADS;
ThreadPool* Engine::computePool = NULL;
unsigned Engine::cThreads = NUM_COMP_THREADS;
std::string Engine::graphFile;
std::string Engine::featuresFile;
std::string Engine::outFile;
std::string Engine::layerConfigFile;
std::string Engine::coordserverIp;
std::string Engine::coordserverPort;
IdType Engine::currId = 0;
Lock Engine::lockCurrId;
Lock Engine::lockRecvWaiters;
Cond Engine::condRecvWaitersEmpty;
Lock Engine::lockHalt;
unsigned Engine::nodeId;
Node& Engine::nodeMe;
unsigned Engine::numNodes;
std::map<IdType, unsigned> Engine::recvWaiters;
Barrier Engine::barComp;
std::vector<unsigned> Engine::layerConfig;
std::vector<unsigned> Engine::layerConfigPrefixSum;
unsigned Engine::numFeatsTotal = 0;
unsigned Engine::numLayers = 0;
FeatType *Engine::verticesDataAll = NULL;
FeatType *Engine::ghostVerticesDataAll = NULL;
FeatType *Engine::verticesDataBuf = NULL;
FeatType *Engine::ghostVerticesDataBuf = NULL;
unsigned Engine::iteration = 0;
bool Engine::undirected = false;
bool Engine::halt = false;
double Engine::timeProcess = 0.0;
double Engine::timeInit = 0.0;

/**
 *
 * Initialize the engine with the given command line arguments.
 * 
 */
void
Engine::init(int argc, char *argv[]) {
    printLog(nodeId, "Engine starts initialization...\n");
    timeInit = -getTimer();

    parseArgs(argc, argv);
    
    // Initialize the node manager and communication manager.
    NodeManager::init(ZKHOST_FILE, HOST_FILE);
    CommManager::init();
    nodeId = NodeManager::getNodeId();
    nodeMe = NodeManager::getNode(nodeId);
    numNodes = NodeManager::getNumNodes();
    assert(numNodes <= 256);    // Cluster size limitation.
    outFile += std::to_string(nodeId);

    // Set number of layers and number of features in each layer. Also store the prefix sum of config for offset querying use.
    readLayerConfigFile(layerConfigFile);
    numLayers = layerConfig.size() - 1;

    unsigned configSum = 0;
    for (unsigned& numFeats : layerConfig) {
        layerConfigPrefixSum.push_back(configSum);
        configSum += numFeats;
    }

    numFeatsTotal = configSum;

    // Read in the graph and subscribe vertex global ID topics.
    std::set<IdType> inTopics;
    std::vector<IdType> outTopics;
    readGraphBS(graphFile, inTopics, outTopics);
    printGraphMetrics();

    // Create the global contiguous memory for vertices' data, according to the given layer config and number of local vertices.
    // Create the global contiguous memory for ghost vertices' data similarly.
    verticesDataAll = new FeatType[configSum * graph.getNumLocalVertices() * sizeof(FeatType)];
    ghostVerticesDataAll = new FeatType[configSum * graph.getNumGhostVertices() * sizeof(FeatType)];
    verticesDataBuf = new FeatType[layerConfig[0] * graph.getNumLocalVertices() * sizeof(FeatType)];
    ghostVerticesDataBuf = new FeatType[layerConfig[0] * graph.getNumGhostVertices() * sizeof(FeatType)];

    // Set a local index for all ghost vertices along the way. This index is used for indexing within the ghost data arrays.
    IdType ghostCount = 0;
    for (auto it = graph.getGhostVertices().begin(); it != graph.getGhostVertices().end(); ++it)
        it->second.setLocalId(ghostCount++);

    // Read in initial features (for the 0th layer) from the features file.
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
    printLog(nodeId, "Created %u computation threads.\n", cThreads);

    CommManager::flushControl();
    CommManager::flushData();

    // Create data communicators thread pool.
    dataPool = new ThreadPool(dThreads);
    dataPool->createPool();
    printLog(nodeId, "Created %u data communicator threads.\n", dThreads);

    // Compact the graph.
    graph.compactGraph();

    timeInit += getTimer();
    printLog(nodeId, "Engine initialization complete.\n");

    // Make sure all nodes finish initailization.
    NodeManager::barrier(INIT_BARRIER);
}



/**
 *
 * Whether I am the master mode or not.
 * 
 */
bool
Engine::master() {
    return NodeManager::amIMaster();
}


/**
 *
 * Run the engine with the given vertex program (whose `update()` member function is customized).
 * Will start a bunch of worker threads and a bunch of data communicator threads.
 * 
 */
void
Engine::run() {
    
    // Make sure engines on all machines start running.
    NodeManager::barrier(RUN_BARRIER); 
    printLog(nodeId, "Engine starts running...\n");

    timeProcess = -getTimer();

    // Change phase to processing.
    NodeManager::startProcessing();

    // Set initial conditions.
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
    if (master())
        printEngineMetrics();
}


/**
 *
 * Write output stuff to the tmp directory for every local vertex.
 * 
 */
void
Engine::output() {
    std::ofstream outStream(outFile.c_str());

    if (!outStream.good())
        printLog(nodeId, "Cannot open output file: %s [Reason: %s]\n", outFile.c_str(), std::strerror(errno));

    assert(outStream.good());

    for (Vertex& v : graph.getVertices()) {
        outStream << v.getGlobalId() << ": ";
        FeatType *dataAllPtr = vertexDataAllPtr(v.getLocalId(), 0);
        unsigned offset = 0;
        for (unsigned& numFeats : layerConfig) {
            for (unsigned i = 0; i < numFeats; ++i)
                outStream << dataAllPtr[offset++] << " ";
            outStream << "| ";
        }
        outStream << std::endl;
    }
}


/**
 *
 * Destroy the engine.
 * 
 */
void
Engine::destroy() {
    NodeManager::destroy();
    CommManager::destroy();
    computePool->destroyPool();
    dataPool->destroyPool();

    lockCurrId.destroy();
    lockRecvWaiters.destroy();
    condRecvWaitersEmpty.destroy();
    lockHalt.destroy();

    delete[] verticesDataAll;
    delete[] ghostVerticesDataAll;
    delete[] verticesDataBuf;
    delete[] ghostVerticesDataBuf;
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
void
Engine::worker(unsigned tid, void *args) {

    // Outer while loop. Looping infinitely, looking for a new task to handle.
    while (1) {

        // Get current vertex that need to be handled.
        lockCurrId.lock();
        IdType lvid = currId++;
        lockCurrId.unlock();

        // All local vertices have been processed. Hit the barrier and wait for next iteration / decide to halt.
        if (lvid >= graph.getNumLocalVertices()) {

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
                printLog(nodeId, "Iteration %u finishes at %.3lf ms.\n", iteration, timeProcess + getTimer());

                //////////////////////////////////
                // Send dataBuf to lambda HERE. //
                //////////////////////////////////
                
                LambdaComm lambdaComm(verticesDataBuf, nodeMe.ip, 65431, graph.getNumLocalVertices(), getNumFeats(), 2, 1);
                
                // Create and launch the sender & receiver workers.
                std::thread t([&] {
                    lambdaComm.run();
                });
                t.detach();

                // Trigger a request towards the coordicate server. Wait until the request completes.
                std::thread t2([&] {
                    lambdaComm.requestLambdas(coordserverIp, coordserverPort, iteration);
                });
                t2.join();

                //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                // Buffer receives results from lambda and should be resized according to the number of features in the next layer. //
                //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

                printLog(nodeId, "All lambdas finished. Waiting on new iteration...\n");

                // Step forward to a new iteration. 
                ++iteration;

                // Flush the returned values from lambda to dataAll.
                for (IdType id = 0; id < graph.getNumLocalVertices(); ++id)
                    memcpy(vertexDataAllPtr(id, getDataAllOffset()), vertexDataBufPtr(id), getNumFeats() * sizeof(FeatType));

                // Wait for all remote schedulings sent by me to be handled.
                lockRecvWaiters.lock();
                if (!recvWaiters.empty())
                    condRecvWaitersEmpty.wait();
                lockRecvWaiters.unlock();

                //## Global Iteration barrier. ##//
                NodeManager::barrier(LAYER_BARRIER);

                // Ghost vertices' flush must happen after all new ghost values received, i.e., global barrier crossed.
                for (IdType id = 0; id < graph.getNumGhostVertices(); ++id)
                    memcpy(ghostVertexDataAllPtr(id, getDataAllOffset()), ghostVertexDataBufPtr(id), getNumFeats() * sizeof(FeatType));

                // There is a new layer ahead, please start a new iteration.
                if (iteration < numLayers) {
                    printLog(nodeId, "Starting a new iteration %u at %.3lf ms...\n", iteration, timeProcess + getTimer());

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

        // Doing the aggregation.
        Vertex& v = graph.getVertex(lvid);
        aggregateFromNeighbors(lvid);

        // If there are any remote edges, should send this vid to others for their ghost's update.
        for (unsigned i = 0; i < v.getNumOutEdges(); ++i) {
            if (v.getOutEdge(i).getEdgeLocation() == REMOTE_EDGE_TYPE) {
                IdType gvid = graph.localToGlobalId[lvid];

                // Record that this vid gets broadcast in this iteration. Should wait for its corresponding respond.
                lockRecvWaiters.lock();
                recvWaiters[gvid] = numNodes;
                lockRecvWaiters.unlock();

                FeatType *dataPtr = vertexDataBufPtr(lvid);
                CommManager::dataPushOut(gvid, dataPtr, getNumFeats() * sizeof(FeatType));

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
void
Engine::dataCommunicator(unsigned tid, void *args) {
    IdType topic;
    FeatType *msgbuf = (FeatType *) new char[MAX_MSG_SIZE];

    // While loop, looping infinitely to get the next message.
    while (1) {

        // No message in queue.
        if (!CommManager::dataPullIn(&topic, msgbuf, MAX_MSG_SIZE)) {

            // Computation workers done their work, so communicator goes to death as well.
            if (halt) {
                delete[] msgbuf;
                return;
            }

        // Pull in the next message, and process this message.
        } else {

            // A normal ghost value broadcast.
            if (0 <= topic && topic < graph.getNumGlobalVertices()) {
                IdType gvid = topic;

                // Update the ghost vertex if it is one of mine.
                if (graph.containsGhostVertex(gvid)) {
                    FeatType *dataBufPtr = ghostVertexDataBufPtr(graph.getGhostVertex(gvid).getLocalId());
                    memcpy(dataBufPtr, msgbuf, getNumFeats() * sizeof(FeatType));
                }

                // Using MAX_IDTYPE - gvid as the receive signal topic for vertex gvid.
                CommManager::dataPushOut(MAX_IDTYPE - gvid, NULL, 0);

            // A respond to a broadcast, and the topic vertex is in my local vertices. I should update the
            // corresponding recvWaiter's value. If waiters become empty, send a signal in case the workers are
            // waiting on it to be empty at the iteration barrier.
            } else if (MAX_IDTYPE >= topic && topic > MAX_IDTYPE - graph.getNumGlobalVertices()) {
                IdType gvid = MAX_IDTYPE - topic;

                if (graph.globalToLocalId.find(gvid) != graph.globalToLocalId.end()) {
                    lockRecvWaiters.lock();
                    assert(recvWaiters.find(gvid) != recvWaiters.end());
                    --recvWaiters[gvid];
                    if (recvWaiters[gvid] == 0) {
                        recvWaiters.erase(gvid);
                        if (recvWaiters.empty())
                            condRecvWaitersEmpty.signal();
                    }
                    lockRecvWaiters.unlock();
                }
            }
        }
    }
}


/**
 *
 * Get number of features in the current layer.
 * 
 */
unsigned
Engine::getNumFeats() {
    return layerConfig[iteration];
}

unsigned
Engine::getNumFeats(unsigned iter) {
    return layerConfig[iter];
}


/**
 *
 * Get the feature starting offset in the current layer.
 * 
 */
unsigned
Engine::getDataAllOffset() {
    return layerConfigPrefixSum[iteration];
}

unsigned
Engine::getDataAllOffset(unsigned iter) {
    return layerConfigPrefixSum[iter];
}


/**
 *
 * Get the data pointer to a local vertex's data in the dataAll area.
 * 
 */
FeatType *
Engine::vertexDataAllPtr(IdType lvid, unsigned offset) {
    return verticesDataAll + lvid * numFeatsTotal + offset;
}


/**
 *
 * Get the data pointer to a ghost vertex's data in the dataAll area.
 * 
 */
FeatType *
Engine::ghostVertexDataAllPtr(IdType lvid, unsigned offset) {
    return ghostVerticesDataAll + lvid * numFeatsTotal + offset;
}


/**
 *
 * Get the data pointer to a local vertex's data in the dataBuf area  (i.e. pointing to values after aggregation).
 * 
 */
FeatType *
Engine::vertexDataBufPtr(IdType lvid) {
    return verticesDataBuf + lvid * getNumFeats();
}

FeatType *
Engine::vertexDataBufPtr(IdType lvid, unsigned numFeats) {
    return verticesDataBuf + lvid * numFeats;
}


/**
 *
 * Get the data pointer to a ghost vertex's data in the dataBuf area (i.e. pointing to values after aggregation).
 * 
 */
FeatType *
Engine::ghostVertexDataBufPtr(IdType lvid) {
    return ghostVerticesDataBuf + lvid * getNumFeats();
}

FeatType *
Engine::ghostVertexDataBufPtr(IdType lvid, unsigned numFeats) {
    return ghostVerticesDataBuf + lvid * numFeats;
}


/**
 *
 * Aggregate numFeats feature values starting from offset from all neighbors (including self). Then write the results to the
 * data buffer area for serialization. The results are to be used for being sent to lambda threads.
 * 
 */
void
Engine::aggregateFromNeighbors(IdType lvid) {
    unsigned numFeats = getNumFeats();
    unsigned offset = getDataAllOffset();

    // Read out data of the current iteration of given vertex.
    FeatType currDataBuf[numFeats];
    FeatType *currDataPtr = vertexDataAllPtr(lvid, offset);
    memcpy(currDataBuf, currDataPtr, numFeats * sizeof(FeatType));

    // Apply normalization factor on the current data.
    Vertex& v = graph.getVertex(lvid);
    for (unsigned i = 0; i < numFeats; ++i)
        currDataBuf[i] *= v.getNormFactor();

    // Aggregate from incoming neighbors.
    for (unsigned i = 0; i < v.getNumInEdges(); ++i) {
        FeatType *otherDataPtr;
        EdgeType normFactor = v.getInEdge(i).getData();

        if (v.getInEdge(i).getEdgeLocation() == LOCAL_EDGE_TYPE)    // Local vertex.
            otherDataPtr = vertexDataAllPtr(v.getSourceVertexLocalId(i), offset);
        else                                                        // Ghost vertex.
            otherDataPtr = ghostVertexDataAllPtr(v.getSourceVertexLocalId(i), offset);

        // TODO: Locks on the data array area is not properly set yet. But does not affect forward prop.
        for (unsigned j = 0; j < numFeats; ++j)
            currDataBuf[j] += (otherDataPtr[j] * normFactor);
    }

    // Write the results to the correct position inside serialization dataBuf area.
    memcpy(vertexDataBufPtr(lvid), currDataBuf, numFeats * sizeof(FeatType));
}


/**
 *
 * Print engine metrics of processing time.
 * 
 */
void
Engine::printEngineMetrics() {
    printLog(nodeId, "Engine METRICS: Initialization time = %.3lf ms.\n", timeInit);
    printLog(nodeId, "Engine METRICS: Processing time = %.3lf ms.\n", timeProcess);
}


/**
 *
 * Print my graph's metrics.
 * 
 */
void
Engine::printGraphMetrics() {
    printLog(nodeId, "Graph METRICS: %u global vertices, %llu global edges, %u local edges.\n",
                     graph.getNumGlobalVertices(), graph.getNumGlobalEdges(), graph.getNumLocalVertices());
}


/**
 *
 * Parse command line arguments.
 * 
 */
void
Engine::parseArgs(int argc, char *argv[]) {
    boost::program_options::options_description desc("Allowed options");
    desc.add_options()
        ("help", "Produce help message")

        ("config", boost::program_options::value<std::string>()->default_value(std::string(DEFAULT_CONFIG_FILE), DEFAULT_CONFIG_FILE), "Config file")
        ("graphfile", boost::program_options::value<std::string>(), "Path to the binary file contatining the edge list")
        ("featuresfile", boost::program_options::value<std::string>(), "Path to the file containing the vertex features")
        ("layerfile", boost::program_options::value<std::string>(), "Layer configuration file")

        ("tmpdir", boost::program_options::value<std::string>(), "Temporary directory")

        ("coordserverip", boost::program_options::value<std::string>(), "The private IP address of the coordination server")
        ("coordserverport", boost::program_options::value<std::string>(), "The port of the listener on the coordination server")

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

    assert(vm.count("layerfile"));
    layerConfigFile = vm["layerfile"].as<std::string>();

    assert(vm.count("tmpdir"));
    outFile = vm["tmpdir"].as<std::string>() + "/output_";  // Still needs to append the node id, after node manager set up.

    assert(vm.count("coordserverip"));
    coordserverIp = vm["coordserverip"].as<std::string>();

    assert(vm.count("coordserverport"));
    coordserverPort = vm["coordserverport"].as<std::string>();

    assert(vm.count("undirected"));
    undirected = (vm["undirected"].as<unsigned>() == 0) ? false : true;

    assert(vm.count("dport"));
    unsigned data_port = vm["dport"].as<unsigned>();
    CommManager::setDataPort(data_port);

    assert(vm.count("cport"));
    unsigned control_port = vm["cport"].as<unsigned>();
    CommManager::setControlPortStart(control_port);

    printLog(nodeId, "Parsed configuration: config = %s, dThreads = %u, cThreads = %u, graphFile = %s,"
                     "featuresFile = %s, undirected = %s, data port set -> %u, control port set -> %u\n",
                     cFile.c_str(), dThreads, cThreads, graphFile.c_str(), featuresFile.c_str(),
                     undirected ? "true" : "false", data_port, control_port);
}


/**
 *
 * Read in the layer configuration file.
 * 
 */
void
Engine::readLayerConfigFile(std::string& layerConfigFileName) {
    std::ifstream infile(layerConfigFileName.c_str());
    if (!infile.good())
        printLog(nodeId, "Cannot open layer configuration file: %s [Reason: %s]\n", layerConfigFileName.c_str(), std::strerror(errno));

    assert(infile.good());

    // Loop through each line.
    std::string line;
    while (!infile.eof()) {
        std::getline(infile, line);
        boost::algorithm::trim(line);

        if (line.length() > 0)
            layerConfig.push_back(std::stoul(line));
    }

    assert(layerConfig.size() > 1);
}


/**
 *
 * Read in the initial features file.
 * 
 */
void
Engine::readFeaturesFile(std::string& featuresFileName) {
    std::ifstream infile(featuresFileName.c_str());
    if (!infile.good())
        printLog(nodeId, "Cannot open feature file: %s [Reason: %s]\n", featuresFileName.c_str(), std::strerror(errno));

    assert(infile.good());

    // Loop through each line.
    unsigned gvid = 0;
    std::string line;
    while (!infile.eof()) {
        std::getline(infile, line);
        boost::algorithm::trim(line);

        std::vector<std::string> splited_strings;
        std::vector<FeatType> feature_vec;

        // Split each line into numbers.
        boost::split(splited_strings, line, boost::is_any_of(", "), boost::token_compress_on);

        for (std::string& substr : splited_strings) {
            if (substr[0] != '\0')      // In case of the null char at the end.
                feature_vec.push_back(std::stof(substr));
        }

        if (feature_vec.size() > 0) {
            assert(feature_vec.size() == layerConfig[0]);

            // Set the vertex's initial values, if it is one of mine local vertices / ghost vertices.
            FeatType *dataPtr = NULL;
            if (graph.containsGhostVertex(gvid))   // Global vertex.
                dataPtr = ghostVertexDataAllPtr(graph.getGhostVertex(gvid).getLocalId(), 0);
            else if (graph.containsVertex(gvid))   // Local vertex.
                dataPtr = vertexDataAllPtr(graph.getVertexByGlobal(gvid).getLocalId(), 0);
            if (dataPtr != NULL)
                memcpy(dataPtr, feature_vec.data(), feature_vec.size() * sizeof(FeatType));

            ++gvid;
        }
    }
}


/**
 *
 * Read in the partition file.
 * 
 */
void
Engine::readPartsFile(std::string& partsFileName, Graph& lGraph) {
    std::ifstream infile(partsFileName.c_str());
    if (!infile.good())
        printLog(nodeId, "Cannot open patition file: %s [Reason: %s]\n", partsFileName.c_str(), std::strerror(errno));

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

        lGraph.appendVertexPartitionId(partId);

        if (partId == nodeId) {
            lGraph.localToGlobalId[lvid] = gvid;
            lGraph.globalToLocalId[gvid] = lvid;

            ++lvid;
        }
        ++gvid;
    }

    lGraph.setNumGlobalVertices(gvid);
    lGraph.setNumLocalVertices(lvid);
}


/**
 *
 * Process an edge read from the binary snap file.
 * 
 */
void
Engine::processEdge(IdType& from, IdType& to, Graph& lGraph, std::set<IdType> *inTopics, std::set<IdType> *oTopics) {
    if (lGraph.getVertexPartitionId(from) == nodeId) {
        IdType lFromId = lGraph.globalToLocalId[from];
        IdType toId;
        EdgeLocationType eLocation;

        if (lGraph.getVertexPartitionId(to) == nodeId) {
            toId = lGraph.globalToLocalId[to];
            eLocation = LOCAL_EDGE_TYPE;
        } else {
            toId = to;
            eLocation = REMOTE_EDGE_TYPE;
            lGraph.getVertex(lFromId).setVertexLocation(BOUNDARY_VERTEX);

            if (oTopics != NULL)
                oTopics->insert(from);
        }

        lGraph.getVertex(lFromId).addOutEdge(OutEdge(toId, eLocation, EdgeType()));
    }

    if (lGraph.getVertexPartitionId(to) == nodeId) {
        IdType lToId = lGraph.globalToLocalId[to];
        IdType fromId;
        EdgeLocationType eLocation;

        if (lGraph.getVertexPartitionId(from) == nodeId) {
            fromId = lGraph.globalToLocalId[from];
            eLocation = LOCAL_EDGE_TYPE;
        } else {
            fromId = from;
            eLocation = REMOTE_EDGE_TYPE;

            if (!lGraph.containsGhostVertex(from))
                lGraph.getGhostVertices()[from] = GhostVertex();
            lGraph.getGhostVertex(from).addOutEdge(lToId);

            if (inTopics != NULL)
                inTopics->insert(from);
        }

        lGraph.getVertex(lToId).addInEdge(InEdge(fromId, eLocation, EdgeType()));
    }
}


/**
 *
 * Set the normalization factors on all edges.
 * 
 */
void
Engine::setEdgeNormalizations() {
    for (Vertex& vertex : graph.getVertices()) {
        unsigned dstDeg = vertex.getNumInEdges() + 1;
        float dstNorm = std::pow(dstDeg, -.5);
        vertex.setNormFactor(dstNorm * dstNorm);
        for (unsigned i = 0; i < vertex.getNumInEdges(); ++i) {
            InEdge& e = vertex.getInEdge(i);
            IdType vid = e.getSourceId();
            if (e.getEdgeLocation() == LOCAL_EDGE_TYPE) {
                unsigned srcDeg = graph.getVertex(vid).getNumInEdges() + 1;
                float srcNorm = std::pow(srcDeg, -.5);
                e.setData(srcNorm * dstNorm);
            } else {
                unsigned ghostDeg = graph.getGhostVertex(vid).getDegree() + 1;
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
void
Engine::findGhostDegrees(std::string& fileName) {
    std::ifstream infile(fileName.c_str(), std::ios::binary);
    if (!infile.good())
        printLog(nodeId, "Cannot open BinarySnap file: %s\n", fileName.c_str());

    assert(infile.good());

    BSHeaderType bsHeader;
    infile.read((char *) &bsHeader, sizeof(bsHeader));

    IdType srcdst[2];
    while (infile.read((char *) srcdst, bsHeader.sizeOfVertexType * 2)) {
        if (srcdst[0] == srcdst[1])
            continue;

        if (graph.containsGhostVertex(srcdst[1]))
            graph.getGhostVertex(srcdst[1]).incrementDegree();
    }
    
    infile.close();
}


/**
 *
 * Read and parse the graph from the graph binary snap file.
 * 
 */
void
Engine::readGraphBS(std::string& fileName, std::set<IdType>& inTopics, std::vector<IdType>& outTopics) {
    
    // Read in the partition file.
    std::string partsFileName = fileName + PARTS_EXT;
    readPartsFile(partsFileName, graph);

    // Initialize the graph based on the partition info.
    graph.getVertices().resize(graph.getNumLocalVertices());
    for (IdType i = 0; i < graph.getNumLocalVertices(); ++i) {
        graph.getVertex(i).setLocalId(i);
        graph.getVertex(i).setGlobalId(graph.localToGlobalId[i]);
        graph.getVertex(i).setVertexLocation(INTERNAL_VERTEX);
        graph.getVertex(i).setGraphPtr(&graph);
    }

    // Read in the binary snap edge file.
    std::string edgeFileName = fileName + EDGES_EXT;
    std::ifstream infile(edgeFileName.c_str(), std::ios::binary);
    if(!infile.good())
        printLog(nodeId, "Cannot open BinarySnap file: %s\n", edgeFileName.c_str());

    assert(infile.good());

    BSHeaderType bSHeader;
    infile.read((char *) &bSHeader, sizeof(bSHeader));
    assert(bSHeader.sizeOfVertexType == sizeof(IdType));

    // Loop through all edges and process them.
    std::set<IdType> oTopics;
    IdType srcdst[2];
    while (infile.read((char *) srcdst, bSHeader.sizeOfVertexType * 2)) {
        if (srcdst[0] == srcdst[1])
            continue;

        processEdge(srcdst[0], srcdst[1], graph, &inTopics, &oTopics);
        if (undirected)
            processEdge(srcdst[1], srcdst[0], graph, &inTopics, &oTopics);
        graph.incrementNumGlobalEdges();
    }

    infile.close();

    // Extra works added.
    graph.setNumGhostVertices(graph.getGhostVertices().size());
    findGhostDegrees(edgeFileName);
    setEdgeNormalizations();

    typename std::set<IdType>::iterator it;
    for (it = oTopics.begin(); it != oTopics.end(); ++it)
        outTopics.push_back(*it);
}
