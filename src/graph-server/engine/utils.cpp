#include "engine.hpp"

#include <omp.h>

#include <algorithm>
#include <boost/algorithm/string/classification.hpp>  // Include boost::for is_any_of.
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/trim.hpp>
#include <boost/program_options.hpp>
#include <boost/program_options/parsers.hpp>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_set>


// ======== Debug utils ========
typedef std::vector<std::vector<unsigned>> opTimes;

// Outputs a string to a file
void outputToFile(std::ofstream &outfile, std::string str) {
    fileMutex.lock();
    outfile.write(str.c_str(), str.size());
    outfile << std::endl << std::flush;
    fileMutex.unlock();
}
// ======== END Debug File utils ========


/**
 *
 * Whether I am the master mode or not.
 *
 */
bool Engine::master() { return nodeManager.amIMaster(); }

void Engine::makeBarrier() { nodeManager.barrier(); }

/**
 *
 * How many epochs to run for
 *
 */
unsigned Engine::getNumEpochs() { return numEpochs; }

/**
 *
 * Add a new epoch time to the list of epoch times
 *
 */
void Engine::addEpochTime(double epochTime) { epochTimes.push_back(epochTime); }

/**
 *
 * How many epochs to run before validation
 *
 */
unsigned Engine::getValFreq() { return valFreq; }

/**
 *
 * Return the ID of this node
 *
 */
unsigned Engine::getNodeId() { return nodeId; }

/**
 * Callback for the LambdaComm to access the NodeManager's epoch update
 * broadcast
 */
void Engine::sendEpochUpdate(unsigned currEpoch) {
    nodeManager.sendEpochUpdate(currEpoch);
}

/**
 *
 * Calculate batch loss and accuracy based on forward predicts and labels
 * locally.
 */
inline void Engine::calcAcc(FeatType *predicts, FeatType *labels,
                            unsigned vtcsCnt, unsigned featDim) {
    float acc = 0.0;
    float loss = 0.0;
    for (unsigned i = 0; i < vtcsCnt; i++) {
        FeatType *currLabel = labels + i * featDim;
        FeatType *currPred = predicts + i * featDim;
        acc += currLabel[argmax(currPred, currPred + featDim)];
        loss -= std::log(currPred[argmax(currLabel, currLabel + featDim)]);
    }
    acc /= vtcsCnt;
    loss /= vtcsCnt;
    printLog(getNodeId(), "batch loss %f, batch acc %f", loss, acc);

    accuracy = acc;
}

/**
 *
 * Write output stuff to the tmp directory for every local vertex.
 * Write engine timing metrics to the logfile.
 *
 */
void Engine::output() {
    std::ofstream outStream(outFile.c_str());
    if (!outStream.good())
        printLog(nodeId, "Cannot open output file: %s [Reason: %s]",
                 outFile.c_str(), std::strerror(errno));

    assert(outStream.good());

    //
    // The following are labels outputing.
    //
    // for (Vertex& v : graph.getVertices()) {
    //     outStream << v.getGlobalId() << ": ";
    //     FeatType *labelsPtr = localVertexLabelsPtr(v.getLocalId());
    //     for (unsigned i = 0; i < layerConfig[numLayers]; ++i)
    //         outStream << labelsPtr[i] << " ";
    //     outStream << std::endl;
    // }

    //
    // The following are timing results outputing.
    //
    // outStream << "I: " << timeInit << std::endl;
    // for (unsigned i = 0; i < numLayers; ++i) {
    //     outStream << "A: " << vecTimeAggregate[i] << std::endl
    //               << "L: " << vecTimeApplyVtx[i] << std::endl
    //               << "G: " << vecTimeScatter[i] << std::endl;
    // }
    // outStream << "B: " << timeBackwardProcess << std::endl;

    char outBuf[1024];
    sprintf(outBuf, "<EM>: Run start time: ");
    outStream << outBuf << std::ctime(&start_time);
    sprintf(outBuf, "<EM>: Run end time: ");
    outStream << outBuf << std::ctime(&end_time);

    sprintf(outBuf, "<EM>: Using %u lambdas", numLambdasForward);
    outStream << outBuf << std::endl;

    sprintf(outBuf, "<EM>: Initialization takes %.3lf ms", timeInit);
    outStream << outBuf << std::endl;

    float avgDenom = static_cast<float>(numSyncEpochs);
    if (pipeline) avgDenom = static_cast<float>(numSyncEpochs * numLambdasForward);

    sprintf(outBuf, "<EM>: Forward:  Time per stage:");
    outStream << outBuf << std::endl;
    for (unsigned i = 0; i < numLayers; ++i) {
        sprintf(outBuf, "<EM>    Aggregation   %2u  %.3lf ms", i,
                vecTimeAggregate[i] / (float)avgDenom);
        outStream << outBuf << std::endl;
        sprintf(outBuf, "<EM>    ApplyVertex   %2u  %.3lf ms", i,
                vecTimeApplyVtx[i] / (float)avgDenom);
        outStream << outBuf << std::endl;
        sprintf(outBuf, "<EM>    Scatter       %2u  %.3lf ms", i,
                vecTimeScatter[i] / (float)avgDenom);
        outStream << outBuf << std::endl;
        sprintf(outBuf, "<EM>    ApplyEdge     %2u  %.3lf ms", i,
                vecTimeApplyEdg[i] / (float)avgDenom);
        outStream << outBuf << std::endl;
    }
    sprintf(outBuf, "<EM>: Total forward-prop time %.3lf ms",
            timeForwardProcess / (float)avgDenom);
    outStream << outBuf << std::endl;

    sprintf(outBuf, "<EM>: Backward: Time per stage:");
    outStream << outBuf << std::endl;
    for (unsigned i = numLayers; i < 2 * numLayers; i++) {
        sprintf(outBuf, "<EM>    Aggregation   %2u  %.3lf ms", i,
                vecTimeAggregate[i] / (float)avgDenom);
        outStream << outBuf << std::endl;
        sprintf(outBuf, "<EM>    ApplyVertex   %2u  %.3lf ms", i,
                vecTimeApplyVtx[i] / (float)avgDenom);
        outStream << outBuf << std::endl;
        sprintf(outBuf, "<EM>    Scatter       %2u  %.3lf ms", i,
                vecTimeScatter[i] / (float)avgDenom);
        outStream << outBuf << std::endl;
        sprintf(outBuf, "<EM>    ApplyEdge     %2u  %.3lf ms", i,
                vecTimeApplyEdg[i] / (float)avgDenom);
        outStream << outBuf << std::endl;
    }
    sprintf(outBuf, "<EM>: Total backward-prop time %.3lf ms",
            timeBackwardProcess);
    outStream << outBuf << std::endl;

    double sum = 0.0;
    for (double &d : epochTimes) sum += d;
    sprintf(outBuf, "<EM>: Average epoch time %.3lf ms",
            sum / (float)epochTimes.size());
    outStream << outBuf << std::endl;
    sprintf(outBuf, "<EM>: Final accuracy %.3lf", accuracy);
    outStream << outBuf << std::endl;
    sprintf(outBuf, "Relaunched Lambda Cnt: %u", resComm->getRelaunchCnt());
    outStream << outBuf << std::endl;

    // Write benchmarking results to log file.
    if (master()) {
        assert(vecTimeAggregate.size() == 2 * numLayers);
        assert(vecTimeApplyVtx.size() == 2 * numLayers);
        assert(pipeline || vecTimeScatter.size() == 2 * numLayers);
        assert(pipeline || vecTimeApplyEdg.size() == 2 * numLayers);
    }
    printEngineMetrics();
}

/**
 *
 * Print engine metrics of processing time.
 *
 */
void Engine::printEngineMetrics() {
    nodeManager.barrier();
    if (master()) {
// for (unsigned i = 0; i < numNodes; i++) {
//     if (nodeId == 0) {
        gtimers.report();

        fprintf(stderr, "[ Node %3u ]  <EM>: Run start time: %s", nodeId, std::ctime(&start_time));
        fprintf(stderr, "[ Node %3u ]  <EM>: Run end time: %s", nodeId, std::ctime(&end_time));
        printLog(nodeId, "<EM>: Backend %s%s", mode == LAMBDA ? "LAMBDA" : (mode == CPU ? "CPU" : "GPU"),
            (mode == LAMBDA ? std::string(":" + lambdaName).c_str() : ""));
        {
            std::string dataset = std::string("<EM>: Dataset: ") + datasetDir + " (";
            for (int i = 0; i < numLayers; i++) {
                dataset += std::to_string(layerConfig[i]) + ", ";
            }
            dataset += std::to_string(layerConfig[numLayers]) +")";
            printLog(nodeId, dataset.c_str());
        }
        printLog(nodeId, "<EM>: staleness: %u", staleness);
        printLog(nodeId, "<EM>: %u sync epochs and %u async epochs",
                numSyncEpochs, numAsyncEpochs);
        printLog(nodeId, "<EM>: Using %u lambdas", numLambdasForward);
        printLog(nodeId, "<EM>: Initialization takes %.3lf ms", timeInit);

        if (false) {
            float avgDenom = static_cast<float>(numSyncEpochs);
            // if (pipeline) avgDenom = static_cast<float>(numSyncEpochs * numLambdasForward);
            printLog(nodeId, "<EM>: Forward:  Time per stage:");
            for (unsigned i = 0; i < numLayers; ++i) {
                printLog(nodeId, "<EM>    Aggregation   %2u  %.3lf ms", i,
                        vecTimeAggregate[i] / (float)(avgDenom));
                printLog(nodeId, "<EM>    ApplyVertex   %2u  %.3lf ms", i,
                        vecTimeApplyVtx[i] / (float)avgDenom);
                printLog(nodeId, "<EM>    Scatter       %2u  %.3lf ms", i,
                        vecTimeScatter[i] / (float)avgDenom);
                printLog(nodeId, "<EM>    ApplyEdge     %2u  %.3lf ms", i,
                        vecTimeApplyEdg[i] / (float)avgDenom);
            }
            printLog(nodeId, "<EM>: Total forward-prop time %.3lf ms",
                    timeForwardProcess / (float)avgDenom);

            printLog(nodeId, "<EM>: Backward: Time per stage:");
            for (unsigned i = numLayers; i < 2 * numLayers; i++) {
                printLog(nodeId, "<EM>    Aggregation   %2u  %.3lf ms", i,
                        vecTimeAggregate[i] / (float)(avgDenom));
                printLog(nodeId, "<EM>    ApplyVertex   %2u  %.3lf ms", i,
                        vecTimeApplyVtx[i] / (float)avgDenom);
                printLog(nodeId, "<EM>    Scatter       %2u  %.3lf ms", i,
                        vecTimeScatter[i] / (float)avgDenom);
                printLog(nodeId, "<EM>    ApplyEdge     %2u  %.3lf ms", i,
                        vecTimeApplyEdg[i] / (float)avgDenom);
            }
            printLog(nodeId, "<EM>: Total backward-prop time %.3lf ms",
                    timeBackwardProcess / (float)avgDenom);

            printLog(nodeId, "<EM>: Final accuracy %.3lf", accuracy);
        }

        printLog(nodeId, "Relaunched Lambda Cnt: %u", resComm->getRelaunchCnt());
    }
    nodeManager.barrier();
// }

    nodeManager.barrier();
    double sum = 0.0;
    for (double &d : epochTimes) sum += d;
    printLog(nodeId, "<EM>: Average  sync epoch time %.3lf ms",
             sum / epochTimes.size());
    nodeManager.barrier();
    printLog(nodeId, "<EM>: Average async epoch time %.3lf ms",
            asyncAvgEpochTime);
}

/**
 *
 * Print my graph's metrics.
 *
 */
void Engine::printGraphMetrics() {
    printLog(nodeId,
             "<GM>: %u global vertices, %llu global edges,\n"
             "\t\t%u local vertices, %llu local in edges, %llu local out edges\n"
             "\t\t%u out ghost vertices, %u in ghost vertices",
             graph.globalVtxCnt, graph.globalEdgeCnt, graph.localVtxCnt,
             graph.localInEdgeCnt, graph.localOutEdgeCnt,
             graph.srcGhostCnt, graph.dstGhostCnt);
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

    ("datasetdir", boost::program_options::value<std::string>(), "Path to the dataset")
    ("featuresfile", boost::program_options::value<std::string>(), "Path to the file containing the vertex features")
    ("layerfile", boost::program_options::value<std::string>(), "Layer configuration file")
    ("labelsfile", boost::program_options::value<std::string>(), "Target labels file")
    ("dshmachinesfile", boost::program_options::value<std::string>(), "DSH machines file")
    ("pripfile", boost::program_options::value<std::string>(), "File containing my private ip")
    ("pubipfile", boost::program_options::value<std::string>(), "File containing my public ip")

    ("tmpdir", boost::program_options::value<std::string>(), "Temporary directory")

    ("dataserverport", boost::program_options::value<unsigned>(), "The port exposing to the lambdas")
    ("weightserverport", boost::program_options::value<unsigned>(), "The port of the listener on the lambdas")
    ("wserveripfile", boost::program_options::value<std::string>(), "The file contains the public IP addresses of the weight server")

    // Default is directed graph!
    ("undirected", boost::program_options::value<unsigned>()->default_value(unsigned(0), "0"), "Graph type is undirected or not")

    ("dthreads", boost::program_options::value<unsigned>(), "Number of data threads")
    ("cthreads", boost::program_options::value<unsigned>(), "Number of compute threads")

    ("dataport", boost::program_options::value<unsigned>(), "Port for data communication")
    ("ctrlport", boost::program_options::value<unsigned>(), "Port start for control communication")
    ("nodeport", boost::program_options::value<unsigned>(), "Port for node manager")

    ("numlambdas", boost::program_options::value<unsigned>()->default_value(unsigned(1), "5"), "Number of lambdas to request at forward")
    ("numEpochs", boost::program_options::value<unsigned>(), "Number of epochs to run")
    ("validationFrequency", boost::program_options::value<unsigned>(), "Number of epochs to run before validation")

    ("MODE", boost::program_options::value<unsigned>(), "0: Lambda, 1: GPU, 2: CPU")
    ("pipeline", boost::program_options::value<bool>(), "0: Sequential, 1: Pipelined")
    ("gnn", boost::program_options::value<std::string>(), "GNN type: [GCN | GAT]")
    ("staleness", boost::program_options::value<unsigned>()->default_value(unsigned(UINT_MAX)),
      "Bound on staleness")
    ("timeout_ratio", boost::program_options::value<unsigned>()->default_value(unsigned(1)),
        "How long to wait for relaunch")
    ;

    boost::program_options::variables_map vm;
    boost::program_options::store(boost::program_options::command_line_parser(argc, argv).options(desc).allow_unregistered().run(), vm);
    boost::program_options::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << std::endl;
        exit(-1);
    }

    assert(vm.count("dthreads"));
    dThreads = vm["dthreads"].as<unsigned>();   // Communicator threads.

    assert(vm.count("cthreads"));
    cThreads = vm["cthreads"].as<unsigned>();   // Computation threads.

    assert(vm.count("datasetdir"));
    datasetDir = vm["datasetdir"].as<std::string>();

    assert(vm.count("featuresfile"));
    featuresFile = vm["featuresfile"].as<std::string>();

    assert(vm.count("layerfile"));
    layerConfigFile = vm["layerfile"].as<std::string>();

    assert(vm.count("labelsfile"));
    labelsFile = vm["labelsfile"].as<std::string>();

    assert(vm.count("dshmachinesfile"));
    dshMachinesFile = vm["dshmachinesfile"].as<std::string>();

    assert(vm.count("pripfile"));
    myPrIpFile = vm["pripfile"].as<std::string>();

    assert(vm.count("tmpdir"));
    outFile = vm["tmpdir"].as<std::string>() + "/output_";  // Still needs to append the node id, after node manager set up.

    assert(vm.count("dataserverport"));
    dataserverPort = vm["dataserverport"].as<unsigned>();

    assert(vm.count("weightserverport"));
    weightserverPort = vm["weightserverport"].as<unsigned>();

    assert(vm.count("wserveripfile"));
    weightserverIPFile = vm["wserveripfile"].as<std::string>();

    assert(vm.count("undirected"));
    undirected = (vm["undirected"].as<unsigned>() == 0) ? false : true;

    assert(vm.count("dataport"));
    unsigned data_port = vm["dataport"].as<unsigned>();
    commManager.setDataPort(data_port);

    assert(vm.count("ctrlport"));
    unsigned ctrl_port = vm["ctrlport"].as<unsigned>();
    commManager.setControlPortStart(ctrl_port);

    assert(vm.count("nodeport"));
    unsigned node_port = vm["nodeport"].as<unsigned>();
    nodeManager.setNodePort(node_port);

    assert(vm.count("numlambdas"));
    numLambdasForward = vm["numlambdas"].as<unsigned>();

    assert(vm.count("numEpochs"));
    numEpochs = vm["numEpochs"].as<unsigned>();

    assert(vm.count("validationFrequency"));
    valFreq = vm["validationFrequency"].as<unsigned>();

    assert(vm.count("MODE"));
    mode = vm["MODE"].as<unsigned>();

    assert(vm.count("pipeline"));
    pipeline = vm["pipeline"].as<bool>();

    assert(vm.count("gnn"));
    std::string gnn_name = vm["gnn"].as<std::string>();
    if (gnn_name == "GCN") {
        gnn_type = GNN::GCN;
    } else if (gnn_name == "GAT") {
        gnn_type = GNN::GAT;
    } else {
        std::cerr << "Unsupported GNN type: " << gnn_name << std::endl;
        exit(-1);
    }

    assert(vm.count("staleness"));
    staleness = vm["staleness"].as<unsigned>();

    assert(vm.count("timeout_ratio"));
    timeoutRatio = vm["timeout_ratio"].as<unsigned>();

    printLog(404, "Parsed configuration: dThreads = %u, cThreads = %u, datasetDir = %s, featuresFile = %s, dshMachinesFile = %s, "
             "myPrIpFile = %s, undirected = %s, data port set -> %u, control port set -> %u, node port set -> %u",
             dThreads, cThreads, datasetDir.c_str(), featuresFile.c_str(), dshMachinesFile.c_str(),
             myPrIpFile.c_str(), undirected ? "true" : "false", data_port, ctrl_port, node_port);
}

/******************************** File utils ********************************/
/**
 *
 * Read in the layer configuration file.
 *
 */
void Engine::readLayerConfigFile(std::string &layerConfigFileName) {
    std::ifstream infile(layerConfigFileName.c_str());
    if (!infile.good())
        printLog(nodeId,
                 "Cannot open layer configuration file: %s [Reason: %s]",
                 layerConfigFileName.c_str(), std::strerror(errno));

    assert(infile.good());

    // Loop through each line.
    std::string line;
    while (!infile.eof()) {
        std::getline(infile, line);
        boost::algorithm::trim(line);

        if (line.length() > 0) layerConfig.push_back(std::stoul(line));
    }

    assert(layerConfig.size() > 1);
}

/**
 *
 * Read in the initial features file.
 *
 */
void Engine::readFeaturesFile(std::string &featuresFileName) {
    bool cache = true;
    if (cache) {
        std::string cacheFeatsFile = datasetDir + "feats" + std::to_string(layerConfig[0]) + "." + std::to_string(nodeId) + ".bin";
        std::ifstream infile(cacheFeatsFile.c_str());
        if (!infile.good()) {
            printLog(nodeId, "No feature cache, loading raw data...",
                     cacheFeatsFile.c_str(), std::strerror(errno));
        } else {
            printLog(nodeId, "Loading feature cache from %s...", cacheFeatsFile.c_str());
            infile.read((char *)forwardVerticesInitData, sizeof(FeatType) * graph.localVtxCnt * layerConfig[0]);
            infile.read((char *)forwardGhostInitData, sizeof(FeatType) * graph.srcGhostCnt * layerConfig[0]);
            infile.close();
            return;
        }
    }
    std::ifstream infile(featuresFileName.c_str());
    if (!infile.good())
        printLog(nodeId, "Cannot open features file: %s [Reason: %s]",
                 featuresFileName.c_str(), std::strerror(errno));
    assert(infile.good());

    FeaturesHeaderType fHeader;
    infile.read((char *)&fHeader, sizeof(FeaturesHeaderType));
    assert(fHeader.numFeatures == layerConfig[0]);

    unsigned gvid = 0;

    unsigned featDim = fHeader.numFeatures;
    std::vector<FeatType> feature_vec;

    feature_vec.resize(featDim);
    while (infile.read(reinterpret_cast<char *>(&feature_vec[0]),
                       sizeof(FeatType) * featDim)) {
        // Set the vertex's initial values, if it is one of my local vertices /
        // ghost vertices.
        if (graph.containsSrcGhostVtx(gvid)) {  // Ghost vertex.
            FeatType *actDataPtr = getVtxFeat(
                forwardGhostInitData,
                graph.srcGhostVtcs[gvid] - graph.localVtxCnt, featDim);
            memcpy(actDataPtr, feature_vec.data(), featDim * sizeof(FeatType));
        } else if (graph.containsVtx(gvid)) {  // Local vertex.
            FeatType *actDataPtr = getVtxFeat(
                forwardVerticesInitData, graph.globaltoLocalId[gvid], featDim);
            memcpy(actDataPtr, feature_vec.data(), featDim * sizeof(FeatType));
        }
        ++gvid;
    }
    infile.close();
    assert(gvid == graph.globalVtxCnt);

    if (cache) {
        std::string cacheFeatsFile = datasetDir + "feats" + std::to_string(layerConfig[0]) + "." + std::to_string(nodeId) + ".bin";
        std::ifstream infile(cacheFeatsFile.c_str());
        if (infile.good()) {
            return;
        }
        std::ofstream outfile(cacheFeatsFile.c_str());
        if (!outfile.good()) {
            printLog(nodeId, "Cannot open output cache file: %s [Reason: %s]",
                     cacheFeatsFile.c_str(), std::strerror(errno));
        }
        outfile.write((char *)forwardVerticesInitData, sizeof(FeatType) * graph.localVtxCnt * layerConfig[0]);
        outfile.write((char *)forwardGhostInitData, sizeof(FeatType) * graph.srcGhostCnt * layerConfig[0]);
        outfile.close();
    }
}

/**
 *
 * Read in the labels file, store the labels in one-hot format.
 *
 */
void Engine::readLabelsFile(std::string &labelsFileName) {
    std::ifstream infile(labelsFileName.c_str());
    if (!infile.good())
        printLog(nodeId, "Cannot open labels file: %s [Reason: %s]",
                 labelsFileName.c_str(), std::strerror(errno));

    assert(infile.good());

    LabelsHeaderType fHeader;
    infile.read((char *)&fHeader, sizeof(LabelsHeaderType));
    assert(fHeader.labelKinds == layerConfig[numLayers]);

    unsigned gvid = 0;

    unsigned lKinds = fHeader.labelKinds;
    unsigned curr;
    FeatType one_hot_arr[lKinds] = {0};

    while (infile.read(reinterpret_cast<char *>(&curr), sizeof(unsigned))) {
        // Set the vertex's label values, if it is one of my local vertices & is
        // labeled.
        if (graph.containsVtx(gvid)) {
            // Convert into a one-hot array.
            assert(curr < lKinds);
            memset(one_hot_arr, 0, lKinds * sizeof(FeatType));
            one_hot_arr[curr] = 1.0;

            FeatType *labelPtr =
                localVertexLabelsPtr(graph.globaltoLocalId[gvid]);
            memcpy(labelPtr, one_hot_arr, lKinds * sizeof(FeatType));
        }

        ++gvid;
    }

    infile.close();
    assert(gvid == graph.globalVtxCnt);
}

void Engine::loadChunks() {
    unsigned vtcsCnt = graph.localVtxCnt;
    for (unsigned cid = 0; cid < numLambdasForward; ++cid) {
        unsigned chunkSize =
            (vtcsCnt + numLambdasForward - 1) / numLambdasForward;
        unsigned lowBound = cid * chunkSize;
        unsigned upBound = std::min(lowBound + chunkSize, vtcsCnt);

        schQueue.push(Chunk { cid, nodeId * numLambdasForward + cid,
                            lowBound, upBound, 0, PROP_TYPE::FORWARD,
                            START_EPOCH + 1, true });
    }

    currEpoch = START_EPOCH;
    // Set the initial bound chunk as epoch 1 layer 0
    minEpoch = currEpoch + 1;
    memset(numFinishedEpoch.data(), 0,
           sizeof(unsigned) * numFinishedEpoch.size());
    memset(numFinishedEpoch.data(), 0,
           sizeof(unsigned) * nodesFinishedEpoch.size());

    finishedChunks = 0;
}

/********************************* SC utils *********************************/
void Engine::verticesPushOut(unsigned receiver, unsigned totCnt,
                             unsigned *lvids, FeatType *inputTensor,
                             unsigned featDim, Chunk &c) {
    zmq::message_t msg(DATA_HEADER_SIZE +
                       (sizeof(unsigned) + sizeof(FeatType) * featDim) *
                           totCnt);
    char *msgPtr = (char *)(msg.data());
    sprintf(msgPtr, NODE_ID_HEADER, receiver);
    msgPtr += NODE_ID_DIGITS;
    unsigned featLayer;
    if (gnn_type == GNN::GCN) { // YIFAN: fix this
        featLayer = c.dir == PROP_TYPE::FORWARD
                  ? c.layer : c.layer - 1;
    } else if (gnn_type == GNN::GAT) {
        featLayer = c.layer - 1;
    }
    populateHeader(msgPtr, nodeId, totCnt, featDim, featLayer, c.dir);
    msgPtr += sizeof(unsigned) * 5;

    for (unsigned i = 0; i < totCnt; ++i) {
        *(unsigned *)msgPtr = graph.localToGlobalId[lvids[i]];
        msgPtr += sizeof(unsigned);
        FeatType *dataPtr = getVtxFeat(inputTensor, lvids[i], featDim);
        memcpy(msgPtr, dataPtr, sizeof(FeatType) * featDim);
        msgPtr += sizeof(FeatType) * featDim;
    }
    commManager.rawMsgPushOut(msg);
}
/********************************* AE utils *********************************/
// reshape vtcs tensor to edgs tensor. Each element in edgsTensor is a reference
// to a vertex feature. Both src vtx features and dst vtx features included in
// edgsTensor. [srcV Feats (local inEdge cnt); dstV Feats (local inEdge cnt)]
FeatType **Engine::srcVFeats2eFeats(FeatType *vtcsTensor, FeatType *ghostTensor,
                                    unsigned vtcsCnt, unsigned featDim) {
    FeatType **eVtxFeatsBuf = new FeatType *[2 * graph.localInEdgeCnt];
    FeatType **eSrcVtxFeats = eVtxFeatsBuf;
    FeatType **eDstVtxFeats = eSrcVtxFeats + graph.localInEdgeCnt;

    unsigned long long edgeItr = 0;
    for (unsigned lvid = 0; lvid < graph.localVtxCnt; ++lvid) {
        for (unsigned long long eid = graph.forwardAdj.columnPtrs[lvid];
             eid < graph.forwardAdj.columnPtrs[lvid + 1]; ++eid) {
            unsigned srcVid = graph.forwardAdj.rowIdxs[eid];
            if (srcVid < graph.localVtxCnt) {
                eSrcVtxFeats[edgeItr] = getVtxFeat(vtcsTensor, srcVid, featDim);
            } else {
                eSrcVtxFeats[edgeItr] = getVtxFeat(
                    ghostTensor, srcVid - graph.localVtxCnt, featDim);
            }
            eDstVtxFeats[edgeItr] = getVtxFeat(vtcsTensor, lvid, featDim);
            ++edgeItr;
        }
    }

    return eVtxFeatsBuf;
}

// similar to srcVFeats2eFeats, but based on outEdges of local vertices.
// [dstV Feats (local outEdge cnt); srcV Feats (local outEdge cnt)]
FeatType **Engine::dstVFeats2eFeats(FeatType *vtcsTensor, FeatType *ghostTensor,
                                    unsigned vtcsCnt, unsigned featDim) {
    FeatType **eVtxFeatsBuf = new FeatType *[2 * graph.localOutEdgeCnt];
    FeatType **eSrcVtxFeats = eVtxFeatsBuf;
    FeatType **eDstVtxFeats = eSrcVtxFeats + graph.localOutEdgeCnt;

    unsigned long long edgeItr = 0;
    for (unsigned lvid = 0; lvid < graph.localVtxCnt; ++lvid) {
        for (unsigned long long eid = graph.backwardAdj.rowPtrs[lvid];
             eid < graph.backwardAdj.rowPtrs[lvid + 1]; ++eid) {
            unsigned srcVid = graph.backwardAdj.columnIdxs[eid];
            if (srcVid < graph.localVtxCnt) {
                eSrcVtxFeats[edgeItr] = getVtxFeat(vtcsTensor, srcVid, featDim);
            } else {
                eSrcVtxFeats[edgeItr] = getVtxFeat(
                    ghostTensor, srcVid - graph.localVtxCnt, featDim);
            }
            eDstVtxFeats[edgeItr] = getVtxFeat(vtcsTensor, lvid, featDim);
            ++edgeItr;
        }
    }

    return eVtxFeatsBuf;
}

unsigned Engine::getAbsLayer(const Chunk &chunk) {
    return chunk.dir == PROP_TYPE::FORWARD
            ? (chunk.layer)
            : (2 * numLayers - 1 - chunk.layer);
}

/******************************** Layer utils ********************************/
Chunk Engine::incLayerGCN(const Chunk &chunk) {
    Chunk nChunk = chunk;
    if (nChunk.dir == PROP_TYPE::FORWARD) { // Forward pass
        nChunk.layer++;
        if (nChunk.layer == numLayers) { // Forward: last layer
            nChunk.dir = PROP_TYPE::BACKWARD; // Switch dir
            // Merge the forward and backward pass of the last layer
            nChunk.layer--;
        }
    } else { // Backward pass
        if (nChunk.layer == 0) { // Epoch ends. Switch dir and inc epoch
            nChunk.dir = PROP_TYPE::FORWARD;
            nChunk.epoch++;
        } else {
            nChunk.layer--;
        }
    }
    return nChunk;
}

Chunk Engine::incLayerGAT(const Chunk &chunk) {
    Chunk nChunk = chunk;
    if (nChunk.dir == PROP_TYPE::FORWARD) { // Forward
        nChunk.layer++;
    } else { // Backward
        if (nChunk.layer == 0) { // Epoch ends. Switch dir and inc epoch
            nChunk.dir = PROP_TYPE::FORWARD;
            nChunk.vertex = true; // reset for a new epoch
            nChunk.epoch++;
        } else {
            nChunk.layer--;
        }
    }
    return nChunk;
}

bool Engine::isLastLayer(const Chunk &chunk) {
    return chunk.dir == PROP_TYPE::BACKWARD &&
           chunk.layer == 0 && chunk.vertex;
}
