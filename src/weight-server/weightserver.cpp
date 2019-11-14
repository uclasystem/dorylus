#include "weightserver.hpp"


std::mutex term_mutex, update_mutex;
std::condition_variable cv;
bool finished = false;


static std::vector<ServerWorker *> workers;
static std::vector<std::thread *> worker_threads;
static std::ofstream outfile;

// set true to write weights to `output_0` for correctness checking.
static bool checkCorrectnessFlag = true;

#define NUM_LISTENERS 5


/** Logging utility. */
void
WeightServer::serverLog(std::string info) {
    std::string msgBase = master ? "[ MASTER ] " : "[ WORKER ] ";
    std::cout << msgBase << info << std::endl;
}


/**
 *
 * Weightserver constructor & destructor.
 *
 */
WeightServer::WeightServer(std::string& weightServersFile, std::string& myPrIpFile,
                           unsigned _listenerPort, std::string& configFileName,
                           unsigned _serverPort, std::string& tmpFileName)
    : ctx(1), frontend(ctx, ZMQ_ROUTER), backend(ctx, ZMQ_DEALER), count(0),
      listenerPort(_listenerPort), numLambdas(0), lambdaRecved(0),
      dataCtx(1), publisher(dataCtx, ZMQ_PUB), subscriber(dataCtx, ZMQ_SUB),
      serverPort(_serverPort) {
    // Hardcoding adam to false for right now
    adam = true;

    // Read the dsh file to get info about all weight server nodes.
    initializeWeightServerComms(weightServersFile, myPrIpFile);

    // Set output file name.
    tmpFileName += std::to_string(nodeId);
    outfile.open(tmpFileName, std::fstream::out);
    assert(outfile.good());

    // Read in layer configurations and initialize weight matrices.
    initializeWeightMatrices(configFileName);

    if (adam) 
        adamOpt=AdamOptimizer(LEARNING_RATE,dims);

    // Send weight matrix info to all servers and wait for ack.
    distributeWeightMatrices();
}

WeightServer::~WeightServer() {
    std::cout << "[SHUTDOWN] Deleting workers" << std::endl;
    // Delete allocated resources.
    for (int i = 0; i < NUM_LISTENERS; ++i) {
        delete workers[i];
        delete worker_threads[i];
    }

    for (Matrix& mat : weightMats)
        delete[] mat.getData();

    std::cout << "[SHUTDOWN] Closing ZMQ" << std::endl;
    frontend.close();
    backend.close();
    ctx.close();

    publisher.close();
    subscriber.close();
    dataCtx.close();
}


/**
 *
 * Runs the weightserver, start a bunch of worker threads and create a proxy through frontend
 * to backend.
 *
 */
void
WeightServer::run() {
    char host_port[50];

    sprintf(host_port, "tcp://*:%u", listenerPort);
    std::cout << "Binding weight server to " << host_port << "..." << std::endl;
    frontend.bind(host_port);
    backend.bind("inproc://backend");

    std::vector<ServerWorker *> workers;
    std::vector<std::thread *> worker_threads;
    WeightServer& me = *this;
    for (int i = 0; i < NUM_LISTENERS; ++i) {
        workers.push_back(new ServerWorker(ctx, me, weightMats, updateMats, numLambdas, lambdaRecved));
        worker_threads.push_back(new std::thread(std::bind(&ServerWorker::work, workers[i])));
        worker_threads[i]->detach();
    }

    try {
        zmq::proxy(static_cast<void *>(frontend), static_cast<void *>(backend), nullptr);
    } catch (std::exception& ex) { /** Context termintated. */ }
}


/**
 *
 * Apply the updates in queue.
 *
 */
void WeightServer::applyUpdate(unsigned layer) {
    Timer updateTimer;
    updateTimer.start();
    std::lock_guard<std::mutex> lk(servers_updates_mutex);
    // Master code.
    if (master) {
        // For all nodes.
        for (unsigned i = 0; i < allNodeIps.size() - 1; ++i) {
            // Recv update info from other weight servers and aggregate.
            zmq::message_t updateMsg;
            subscriber.recv(&updateMsg);

            FeatType *updateSum = updateMats[layer].getData();
            FeatType *updateNew = (FeatType *) updateMsg.data();
            for (unsigned u = 0; u < updateMats[layer].getNumElemts(); ++u)
                updateSum[u] += updateNew[u];
        }
        
        FeatType *weightData = weightMats[layer].getData();
        FeatType *updateSum = updateMats[layer].getData();
        if (adam) {
            adamOpt.update(layer,weightData,updateSum);
        }
        else{
            // Once all updates have been aggregated, apply to the weights matrices.
            for (unsigned u = 0; u < weightMats[layer].getNumElemts(); ++u)
                weightData[u] -=  LEARNING_RATE * updateSum[u];
        }

        // Send out the updated weights.
        Matrix& weightMat = weightMats[layer];
        zmq::message_t weightDataMsg(weightMat.getDataSize());
        std::memcpy(weightDataMsg.data(), weightMat.getData(), weightMat.getDataSize());

        publisher.send(weightDataMsg);

        // Wait for all ACK messages.
        zmq::message_t inMsg;
        unsigned acksNeeded = allNodeIps.size() - 1;
        while (acksNeeded > 0) {
            subscriber.recv(&inMsg);

            unsigned msgType;
            std::memcpy(&msgType, inMsg.data(), inMsg.size());
            if (msgType == CTRL_MSG::ACK)
                acksNeeded--;
        }

        servers_updates_done=true;
        servers_updates_cv.notify_all();
        serverLog("Finished updating the weights.");

        //
        // Uncomment below to write updated weights results to `output_0` for correctness checking.
        //
        if (checkCorrectnessFlag) {
            float tmp = 0.0;
            for (unsigned i = 0; i < updateMats[layer].getNumElemts(); i++) {
                tmp += std::fabs(updateMats[layer].getData()[i]);
            }
            
            std::cout << "Layer " << layer << " Weight Grad Agg: " << tmp << " Max element: " << *(std::max_element(updateMats[layer].getData(), updateMats[layer].getData() + updateMats[layer].getNumElemts())) << " Min element: " << *(std::min_element(updateMats[layer].getData(), updateMats[layer].getData() + updateMats[layer].getNumElemts())) << std::endl;
            // std::cout << "[";
            // for (unsigned i = 0; i < 10; i++) {
            //     std::cout << updateMats[layer].getData()[i] << " ";
            // }
            // std::cout << "]\n";
        }

    // Worker code.
    } else {

        // Send all updated weight matrices to the master for aggregation.
        for (unsigned i = 0; i < updateMats.size(); ++i) {
            Matrix& updateMat = updateMats[i];
            zmq::message_t updateDataMsg(updateMat.getDataSize());
            std::memcpy(updateDataMsg.data(), updateMat.getData(), updateMat.getDataSize());

            if (i == updateMats.size() - 1)
                publisher.send(updateDataMsg);
            else
                publisher.send(updateDataMsg, ZMQ_SNDMORE);
        }

        // Wait for the master to reply with the aggregated and averaged weight values.
        for (Matrix& weightMat : weightMats) {
            zmq::message_t updatedWeightMsg;
            subscriber.recv(&updatedWeightMsg);

            assert(weightMat.getDataSize() == updatedWeightMsg.size());

            // If there are no errors, copy the new data into the weight matrix.
            std::memcpy(weightMat.getData(), updatedWeightMsg.data(), weightMat.getDataSize());
        }

        // Send back confirm ACK message.
        unsigned msgType = CTRL_MSG::ACK;
        zmq::message_t ackMsg(sizeof(unsigned));
        std::memcpy(ackMsg.data(), &msgType, sizeof(unsigned));
        publisher.send(ackMsg);

        servers_updates_done=true;
        servers_updates_cv.notify_all();
        serverLog("All workers weights updated.");
    }

    updateTimer.stop();
    outfile << "U: " << updateTimer.getTime() << std::endl;     // Output timing results.

    // Clear the update buffer.
    for (unsigned l = 0; l < updateMats.size(); ++l) {
        float *updateData = updateMats[l].getData();
        for (unsigned u = 0; u < updateMats[l].getNumElemts(); ++u)
            updateData[u] = 0.;
    }

    // Reset number of lambdas.
    lambdaRecved = 0;
}

void WeightServer::applyUpdates() {

    Timer updateTimer;
    updateTimer.start();

    // Master code.
    if (master) {

        // For all nodes.
        for (unsigned i = 0; i < allNodeIps.size() - 1; ++i) {

            // For all layers.
            for (unsigned l = 0; l < updateMats.size(); ++l) {

                // Recv update info from other weight servers and aggregate.
                zmq::message_t updateMsg;
                subscriber.recv(&updateMsg);

                FeatType *updateSum = updateMats[l].getData();
                FeatType *updateNew = (FeatType *) updateMsg.data();
                for (unsigned u = 0; u < updateMats[l].getNumElemts(); ++u)
                    updateSum[u] += updateNew[u];
            }
        }

        // If adam is enabled, apply the momentum and decay computation
        if (adam) {
            for (unsigned i = 0; i < updateMats.size(); ++i) {
            }
        }

        // Once all updates have been aggregated, apply to the weights matrices.
        for (unsigned l = 0; l < weightMats.size(); ++l) {
            FeatType *weightData = weightMats[l].getData();
            FeatType *updateSum = updateMats[l].getData();
            for (unsigned u = 0; u < weightMats[l].getNumElemts(); ++u)
                weightData[u] -= updateSum[u];
        }

        // Send out the updated weights.
        for (unsigned l = 0; l < weightMats.size(); ++l) {
            Matrix& weightMat = weightMats[l];
            zmq::message_t weightDataMsg(weightMat.getDataSize());
            std::memcpy(weightDataMsg.data(), weightMat.getData(), weightMat.getDataSize());

            if (l == weightMats.size() - 1)
                publisher.send(weightDataMsg);
            else
                publisher.send(weightDataMsg, ZMQ_SNDMORE);
        }

        // Wait for all ACK messages.
        zmq::message_t inMsg;
        unsigned acksNeeded = allNodeIps.size() - 1;
        while (acksNeeded > 0) {
            subscriber.recv(&inMsg);

            unsigned msgType;
            std::memcpy(&msgType, inMsg.data(), inMsg.size());
            if (msgType == CTRL_MSG::ACK)
                acksNeeded--;
        }

        serverLog("Finished updating the weights.");

        //
        // Uncomment below to write updated weights results to `output_0` for correctness checking.
        //
        if (checkCorrectnessFlag) {
            for (Matrix& mat : weightMats)
                outfile << mat.str() << std::endl;
        }

    // Worker code.
    } else {

        // Send all updated weight matrices to the master for aggregation.
        for (unsigned i = 0; i < updateMats.size(); ++i) {
            Matrix& updateMat = updateMats[i];
            zmq::message_t updateDataMsg(updateMat.getDataSize());
            std::memcpy(updateDataMsg.data(), updateMat.getData(), updateMat.getDataSize());

            if (i == updateMats.size() - 1)
                publisher.send(updateDataMsg);
            else
                publisher.send(updateDataMsg, ZMQ_SNDMORE);
        }

        // Wait for the master to reply with the aggregated and averaged weight values.
        for (Matrix& weightMat : weightMats) {
            zmq::message_t updatedWeightMsg;
            subscriber.recv(&updatedWeightMsg);

            assert(weightMat.getDataSize() == updatedWeightMsg.size());

            // If there are no errors, copy the new data into the weight matrix.
            std::memcpy(weightMat.getData(), updatedWeightMsg.data(), weightMat.getDataSize());
        }

        // Send back confirm ACK message.
        unsigned msgType = CTRL_MSG::ACK;
        zmq::message_t ackMsg(sizeof(unsigned));
        std::memcpy(ackMsg.data(), &msgType, sizeof(unsigned));
        publisher.send(ackMsg);

        serverLog("All workers weights updated.");
    }

    updateTimer.stop();
    outfile << "U: " << updateTimer.getTime() << std::endl;     // Output timing results.

    // Clear the update buffer.
    for (unsigned l = 0; l < updateMats.size(); ++l) {
        float *updateData = updateMats[l].getData();
        for (unsigned u = 0; u < updateMats[l].getNumElemts(); ++u)
            updateData[u] = 0.;
    }

    // Reset number of lambdas.
    lambdaRecved = 0;
}


/**
 *
 * Use dsh file to open sockets to other weight servers.
 *
 */
void
WeightServer::initializeWeightServerComms(std::string& weightServersFile, std::string& myPrIpFile) {
    std::string myIp;
    std::string masterIp = parseNodeConfig(weightServersFile, myPrIpFile, myIp);
    if (master)
        serverLog("Initializing nodes...");

    // Everyone needs to bind to a publisher socket.
    // Need to use the private IP because of port restrictions.
    publisher.setsockopt(ZMQ_SNDHWM, 0);    // Set no limit on message queue.
    publisher.setsockopt(ZMQ_RCVHWM, 0);
    char myHostPort[50];
    sprintf(myHostPort, "tcp://%s:%u", myIp.c_str(), serverPort);
    publisher.bind(myHostPort);

    subscriber.setsockopt(ZMQ_SNDHWM, 0);
    subscriber.setsockopt(ZMQ_RCVHWM, 0);
    if (master) {       // If you are the master, subscribe all other nodes.
        for (std::string& ipStr : allNodeIps) {
            if (ipStr != myIp) {
                char hostPort[50];
                sprintf(hostPort, "tcp://%s:%u", ipStr.c_str(), serverPort);
                subscriber.connect(hostPort);
            }
        }
    } else {            // If you are a worker, just connect to master.
        char hostPort[50];
        sprintf(hostPort, "tcp://%s:%u", masterIp.c_str(), serverPort);
        subscriber.connect(hostPort);
    }
    subscriber.setsockopt(ZMQ_SUBSCRIBE, NULL, 0);

    // Subscribe process.
    if (master) {
        unsigned remaining = allNodeIps.size() - 1;

        // Keep polling until all workers reponsd
        while (remaining > 0) {
            zmq::message_t outMsg1(sizeof(unsigned));  // Send msg 1.
            unsigned ctrlMsg = CTRL_MSG::MASTERUP;
            std::memcpy(outMsg1.data(), &ctrlMsg, sizeof(unsigned));
            publisher.ksend(outMsg1);

            std::this_thread::sleep_for(std::chrono::milliseconds(500));

            zmq::message_t inMsg;
            while (subscriber.krecv(&inMsg, ZMQ_DONTWAIT)) {    // Wait on ALL msg 2.
                unsigned inCtrlMsg;
                std::memcpy(&inCtrlMsg, inMsg.data(), inMsg.size());
                if (inCtrlMsg == CTRL_MSG::WORKERUP) {
                    --remaining;
                }
            }
        }

        zmq::message_t outMsg2(sizeof(unsigned));   // Send msg 3 (init finished).
        unsigned doneMsg = CTRL_MSG::INITDONE;
        std::memcpy(outMsg2.data(), &doneMsg, sizeof(unsigned));
        publisher.send(outMsg2);

    } else {

        zmq::message_t inMsg;   // Recv msg 1.
        while (subscriber.recv(&inMsg)) {
            unsigned msgType;
            std::memcpy(&msgType, inMsg.data(), inMsg.size());

            if (msgType == CTRL_MSG::MASTERUP)
                break;
        }

        zmq::message_t outMsg(sizeof(unsigned));;  // Send msg 2 (ack).
        unsigned outMsgType = CTRL_MSG::WORKERUP;
        std::memcpy(outMsg.data(), &outMsgType, sizeof(unsigned));
        publisher.send(outMsg);

        while (subscriber.recv(&inMsg)) {
            unsigned doneMsg;
            std::memcpy(&doneMsg, inMsg.data(), sizeof(unsigned));
            if (doneMsg == CTRL_MSG::INITDONE)
                break;
        }
    }

    if (master)
        serverLog("All weight servers connected.");
}

std::string
WeightServer::parseNodeConfig(std::string& weightServersFile, std::string& myPrIpFile, std::string& myIp) {
    std::ifstream ipFile(myPrIpFile);
    assert(ipFile.good());

    std::getline(ipFile, myIp);
    ipFile.close();

    std::ifstream dshFile(weightServersFile);
    std::string line, masterIp;
    while (std::getline(dshFile, line)) {
        boost::algorithm::trim(line);
        if (line.length() > 0) {
            std::string ip = line.substr(line.find('@') + 1);

            // Set first node in file as master.
            if (ip == myIp) {
                nodeId = allNodeIps.size();
                master = (nodeId == 0);
            }

            // Even if this is not your IP, it is the master IP.
            if (allNodeIps.empty())
                masterIp = ip;

            allNodeIps.push_back(ip);
        }
    }

    return masterIp;
}


/**
 *
 * Read in layer configurations.
 *
 */
void
WeightServer::initializeWeightMatrices(std::string& configFileName) {

    // Read the layer config file. Each line is a number of features.
    std::ifstream infile(configFileName.c_str());
    if (!infile.good())
        fprintf(stderr, "[ERROR] Cannot open layer configuration file: %s [Reason: %s]\n", configFileName.c_str(), std::strerror(errno));

    assert(infile.good());

    // Loop through each line.
    std::string line;
    while (!infile.eof()) {
        std::getline(infile, line);
        boost::algorithm::trim(line);

        if (line.length() > 0)
            dims.push_back(std::stoul(line));
    }

    // Assert there is at least one layer (input -> output).
    assert(dims.size() > 1);

    // If master node, initialize the weight matrices according to the layer config.
    if (master) {
        for (unsigned u = 0; u < dims.size() - 1; ++u) {
            // Hardcoding this to xavier init for now. Eventually need to make it
            // configurable
            Matrix w = xavierInitialization(dims[u], dims[u+1]);
            weightMats.push_back(w);

            // Initialize layer biases
            // TODO:
            //  Make this configurable based on whether or not a bias matrix is requested
            //  for a NN module
            Matrix b = initBias(dims[u+1]);
            biases.push_back(b);
        }

        for (unsigned u = 0; u < weightMats.size(); ++u)
            serverLog("Layer " + std::to_string(u) + " - Weights: " + weightMats[u].shape());

        // // for checking correctness
        // if (checkCorrectnessFlag) {
        //     for (Matrix& mat : weightMats) {
        //         outfile << mat.str() << std::endl;
        //     }
        // }
    }

    // For all nodes, initialize empty update matrices buffers.
    for (unsigned u = 0; u < dims.size() - 1; ++u) {
        unsigned dataSize = dims[u] * dims[u + 1];
        float *dptr = new float[dataSize];

        for (unsigned ui = 0; ui < dataSize; ++ui)
            dptr[ui] = 0.;

        updateMats.push_back(Matrix(dims[u], dims[u + 1], dptr));
    }
}

/**
 *
 * Used for weights init when tanh or some other symmetric
 * activation is being used
 *
 */
Matrix
WeightServer::xavierInitialization(unsigned dim1, unsigned dim2) {
    int t= 1573771109;
    printf("Time %u\n",t);
    std::default_random_engine dre(t);
    std::uniform_real_distribution<float> dist(-1, 1);

    unsigned dataSize = dim1 * dim2;
    float *dptr = new float[dataSize];

    for (unsigned ui = 0; ui < dataSize; ++ui)
        dptr[ui] = dist(dre);

    float normFactor = std::sqrt(6.0 / (float (dim1 + dim2)));
    for (unsigned ui = 0; ui < dataSize; ++ui)
        dptr[ui] *= normFactor;
    return Matrix(dim1, dim2, dptr);
}

/**
 *
 * Used for weights init when the ReLU or some other asymmetric
 * activation function is used
 *
 */
Matrix
WeightServer::kaimingInitialization(unsigned dim1, unsigned dim2) {
    std::default_random_engine dre(8888);
    std::normal_distribution<float> dist(0, 1);

    unsigned dataSize = dim1 * dim2;
    FeatType *dptr = new FeatType[dataSize];

    for (unsigned ui = 0; ui < dataSize; ++ui)
        dptr[ui] = dist(dre);

    float normFactor = std::sqrt(2.0 / (float(dim1)));
    for (unsigned ui = 0; ui < dataSize; ++ui)
        dptr[ui] *= normFactor;

    return Matrix(dim1, dim2, dptr);
}

/**
 *
 * Randomly initialize weights
 *
 */
Matrix
WeightServer::randomInitialization(unsigned dim1, unsigned dim2,
                                   float lowerBound, float upperBound) {
    assert(lowerBoud < upperBound);
    std::default_random_engine dre(8888);
    std::uniform_real_distribution<float> dist(lowerBound, upperBound);

    unsigned dataSize = dim1 * dim2;
    FeatType *dptr = new FeatType[dataSize];

    for (unsigned ui = 0; ui < dataSize; ++ui)
        dptr[ui] = dist(dre);

    return Matrix(dim1, dim2, dptr);
}

/**
 *
 * Initialize bias vectors to output size of layer
 *
 */
Matrix
WeightServer::initBias(unsigned dim, float initVal) {
    // TODO:
    //  Generalize implementation of tensors/matrices to include 3-D matrices and vectors
    FeatType* dptr = new FeatType[dim];

    for (unsigned ui = 0; ui < dim; ++ui)
        dptr[ui] = initVal;

    return Matrix(dim, 1, dptr);
}


/**
 *
 * Distribute the weight matrices from the master to the other weight servers
 *
 */
void
WeightServer::distributeWeightMatrices() {

    // Master code.
    if (master) {

        // Master sends all the weight matrices to the worker nodes.
        for (unsigned i = 0; i < weightMats.size(); ++i) {
            Matrix& weights = weightMats[i];

            zmq::message_t weightData(weights.getDataSize());
            std::memcpy((char *) weightData.data(), weights.getData(), weights.getDataSize());

            if (i == weightMats.size() - 1)
                publisher.send(weightData);
            else
                publisher.send(weightData, ZMQ_SNDMORE);
        }

        // Get an ACK from every worker node that they have received the weights.
        zmq::message_t inMsg;
        int acksNeeded = allNodeIps.size() - 1;
        while (acksNeeded > 0) {
            subscriber.recv(&inMsg);

            unsigned msgType;
            std::memcpy(&msgType, inMsg.data(), inMsg.size());
            if (msgType == CTRL_MSG::ACK) {
                acksNeeded--;
            }
        }

    // Worker code.
    } else {

        // Worker receives each weight matrix.
        int more = 0;
        do {
            zmq::message_t weightData;
            subscriber.recv(&weightData);

            char *matxData = new char[weightData.size()];
            std::memcpy(matxData, weightData.data(), weightData.size());

            weightMats.push_back(Matrix(dims[count], dims[count+1], matxData));
            ++count;

            size_t more_size = sizeof(more);
            subscriber.getsockopt(ZMQ_RCVMORE, &more, &more_size);
        } while (more);

        // After all matrices have been received, alert the master.
        unsigned msgType = CTRL_MSG::ACK;
        zmq::message_t ackMsg(sizeof(unsigned));
        std::memcpy(ackMsg.data(), &msgType, sizeof(unsigned));
        publisher.send(ackMsg);
    }

    if (master)
        serverLog("All nodes up to date.");
}


/** Main entrance: Starts a weightserver instance and run. */
int
main(int argc, char *argv[]) {

    // TODO: May need to start using an arg parser like boost.
    assert(argc == 7);
    std::string weightServersFile = argv[1];
    std::string myPrIpFile = argv[2];
    unsigned serverPort = std::atoi(argv[3]);
    unsigned listenerPort = std::atoi(argv[4]);
    std::string configFileName = argv[5];

    // Set output file location. Still needs to append nodeId.
    std::string tmpFileName = std::string(argv[6]) + "/output_";

    WeightServer ws(weightServersFile, myPrIpFile, listenerPort, configFileName, serverPort, tmpFileName);

    // Run in a detached thread because so that we can wait
    // on a condition variable.
    std::thread t([&]{
        ws.run();
    });
    t.detach();

    // Wait for one of the threads to mark the finished bool true
    // then end the main thread.
    std::unique_lock<std::mutex> lk(term_mutex);
    cv.wait(lk, [&]{ return finished; });
    std::cerr << "We are terminating the weight server" << std::endl;

    return 0;
}
