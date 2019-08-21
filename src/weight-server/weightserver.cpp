#include "weightserver.hpp"


std::mutex term_mutex, update_mutex;
std::condition_variable cv;
bool finished = false;


/**
 *
 * Weightserver constructor.
 * 
 */
WeightServer::WeightServer(std::string& weightServersFile, std::string& myPrIpFile, unsigned _listenerPort, std::string& configFileName, unsigned _serverPort)
    : ctx(1), frontend(ctx, ZMQ_ROUTER), backend(ctx, ZMQ_DEALER),
      listenerPort(_listenerPort), numLambdas(20), count(0),
      dataCtx(1), publisher(dataCtx, ZMQ_PUB), subscriber(dataCtx, ZMQ_SUB),
      serverPort(_serverPort) {

	// Read the dsh file to get info about all weight server nodes
	initializeWeightServerComms(weightServersFile, myPrIpFile);

    // Read in layer configurations.
    initializeWeightMatrices(configFileName);

    // TODO: Currently using randomly generated weights.
    if (master) {
        auto seed = 8888;
        std::default_random_engine dre(seed);
        std::uniform_real_distribution<FeatType> dist(-1.5, 1.5);

        for (unsigned u = 0; u < dims.size() - 1; ++u) {
            unsigned dataSize = dims[u] * dims[u + 1];
            FeatType *dptr = new FeatType[dataSize];
            
            for (unsigned ui = 0; ui < dataSize; ++ui)
                dptr[ui] = dist(dre);

            weightMats.push_back(Matrix(dims[u], dims[u + 1], dptr));
        }

        for (unsigned u = 0; u < weightMats.size(); ++u)
            fprintf(stdout, "Layer %u - Weights: %s\n", u, weightMats[u].shape().c_str());
    }
}


/**
 *
 * Runs the weightserver, start a bunch of worker threads and create a proxy through frontend to backend.
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
    for (int i = 0; i < kMaxThreads; ++i) {
        workers.push_back(new ServerWorker(ctx, count, me, weightMats, updates, numLambdas));

        worker_threads.push_back(new std::thread(std::bind(&ServerWorker::work, workers[i])));
        worker_threads[i]->detach();
    }

    try {
        zmq::proxy(static_cast<void *>(frontend), static_cast<void *>(backend), nullptr);
    } catch (std::exception& ex) { /** Context termintated. */ }

    // Delete the workers after the context terminates.
    for (int i = 0; i < kMaxThreads; ++i) {
        delete worker_threads[i];
        delete workers[i];
    }
    for (Matrix& mat : weightMats)
        delete[] mat.getData();
}


/**
 *
 * Apply the updates in queue.
 * 
 */
void WeightServer::applyUpdates() {
    std::lock_guard<std::mutex> update_lock(update_mutex);

    // Use pop() to avoid extra clearing.
}

/**
 *
 * Use dsh file to open sockets to other weight servers
 *
 */
void
WeightServer::initializeWeightServerComms(std::string& weightServersFile,
  std::string& myPrIpFile) {
    std::string myIp;
    std::string masterIp = parseNodeConfig(weightServersFile, myPrIpFile, myIp);
    if (master) {
        std::cout << "Initializing nodes" << std::endl;
    }

    // Everyone needs to bind to a publisher socket
    // Need to use the private IP because of port restrictions
    publisher.setsockopt(ZMQ_SNDHWM, 0);    // no limit on message queue
    publisher.setsockopt(ZMQ_RCVHWM, 0);
    char myHostPort[50];
    sprintf(myHostPort, "tcp://%s:%u", myIp.c_str(), serverPort);
    publisher.bind(myHostPort);

    subscriber.setsockopt(ZMQ_SNDHWM, 0);
    subscriber.setsockopt(ZMQ_RCVHWM, 0);
    // if you are the master connect to all other nodes
    if (master) {
        for (std::string& ipStr : allNodeIps) {
            char hostPort[50];
            sprintf(hostPort, "tcp://%s:%u", ipStr.c_str(), serverPort);
            subscriber.connect(hostPort);
        }
    // if not the master only connect to the master
    } else {
        char hostPort[50];
        sprintf(hostPort, "tcp://%s:%u", masterIp.c_str(), serverPort);
    }

    subscriber.setsockopt(ZMQ_SUBSCRIBE, NULL, 0);

    if (master) {
        unsigned remaining = allNodeIps.size() - 1;

        // Keep polling until all workers reponsd
        while (remaining > 0) {
            zmq::message_t outMsg1;  // Send msg 1
            unsigned msg = CTRL_MSG::MASTERUP;
            *((unsigned*) outMsg1.data()) = msg;
            publisher.ksend(outMsg1);
            std::cout << "[MASTER] Sending msg 1" << std::endl;

            std::this_thread::sleep_for(std::chrono::milliseconds(500));

            zmq::message_t inMsg;
            while (subscriber.krecv(&inMsg, ZMQ_DONTWAIT)) {    // wait on ALL msg 2
                unsigned inCtrlMsg = *((unsigned*) inMsg.data());
                if (inCtrlMsg == CTRL_MSG::WORKERUP) {
                    std::cout << "[MASTER] Recv msg 2" << std::endl;
                    --remaining;
                }
            }

        }

        std::cout << "[MASTER] Seding init finished msg" << std::endl;
        zmq::message_t outMsg2; // Send msg 3 (init finished)
        publisher.send(outMsg2);
    } else {
        // Worker
        std::cout << "[WORKER] Waiting on msg 1" << std::endl;
        zmq::message_t inMsg1;   // Recv msg 1
        subscriber.recv(&inMsg1);

        std::cout << "[WORKER] Msg 1 recvd. Sending ack" << std::endl;
        zmq::message_t outMsg;  // Send msg 2 (ack)
        publisher.send(outMsg);

        std::cout << "[WORKER] Waiting on msg 3" << std::endl;
        zmq::message_t inMsg2;   // Recv msg 3 (init finished)
        subscriber.recv(&inMsg2);
    }

    if (master) {
        std::cout << "All weight servers connected" << std::endl;
    }
}

std::string
WeightServer::parseNodeConfig(std::string& weightServersFile,
    std::string& myPrIpFile, std::string& myIp) {
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

            // Set first node as master
            if (ip == myIp) {
                master = allNodeIps.empty();
            }

            // Even if this is not your IP, it is the master IP
            if (allNodeIps.empty()) {
                masterIp = ip;
            }

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

    assert(dims.size() > 1);
}


/** Main entrance: Starts a weightserver instance and run. */
int
main(int argc, char *argv[]) {
    // TODO:
    //  May need to start using an arg parser like boost
    assert(argc == 5);
	std::string weightServersFile = argv[1];
	std::string myPrIpFile = argv[2];
    unsigned serverPort = std::atoi(argv[3]);
    unsigned listenerPort = std::atoi(argv[4]);
    std::string configFileName = argv[5];

    WeightServer ws(weightServersFile, myPrIpFile, listenerPort,
        configFileName, serverPort);
    
    // Run in a detached thread because so that we can wait
    // on a condition variable.
    std::thread t([&]{
        ws.run();
    });
    t.detach();

    // Wait for one of the threads to mark the finished bool true
    // then end the main thread
    std::unique_lock<std::mutex> lk(term_mutex);
    cv.wait(lk, [&]{ return finished; });
    
    return 0;
}
