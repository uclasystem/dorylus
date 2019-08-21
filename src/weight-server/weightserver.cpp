#include "weightserver.hpp"


std::mutex term_mutex, update_mutex;
std::condition_variable cv;
bool finished = false;

void workerLog(std::string msg) {
    std::cout << "[WORKER] " << msg << std::endl;
}

void masterLog(std::string msg) {
    std::cout << "[MASTER] " << msg << std::endl;
}


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

    // Read in layer configurations and initialize matrices
    initializeWeightMatrices(configFileName);

    // Send weight matrix info to all servers and wait for ack
    distributeWeightMatrices();

    if (master) {
        int layer = 0;
        for (Matrix& mat : weightMats) {
            fprintf(stderr, "[MASTER] Layer %u\n%s\n", ++layer, mat.str().c_str());
        }
    } else {
        int layer = 0;
        for (Matrix& mat : weightMats) {
            fprintf(stderr, "[WORKER] Layer %u\n%s\n", ++layer, mat.str().c_str());
        }
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
    // Grab this lock in case a labmda is fast enough to start updating the
    // weights before we can finish the apply
    std::lock_guard<std::mutex> update_lock(update_mutex);
    if (master) {
        for (unsigned i = 0; i < allNodeIps.size() - 1; ++i) {
            for (unsigned l = 0; l < weightMats.size(); ++l) {
                zmq::message_t updateMsg;
                subscriber.recv(&updateMsg);

                FeatType* updateData = updateMsg.data();
                FeatType* weightData = weightMats[i].getData();
                for (unsigned u = 0; u < weightMats[i].getNumElemts(); ++u) {
                    weightData[u] += updateData[u];
                }
            }
        }

    // Worker code
    } else {
        for (unsigned i = 0; i < weightMats.size(); ++i) {
            Matrix& weightMat = weightMats[i];
            zmq::message_t weightDataMsg(weightMat.getDataSize());
            std::memcpy(weightDataMsg.data(), weightMat.getData(), weightMat.getDataSize());

            if (i == weightMats.size() - 1)
                publisher.send(weightDataMsg);
            else
                publisher.send(weightDataMsg, ZMQ_SNDMORE);
        }
    }
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
            if (ipStr != myIp) {
                char hostPort[50];
                sprintf(hostPort, "tcp://%s:%u", ipStr.c_str(), serverPort);
                
                subscriber.connect(hostPort);
            }
        }
    // if not the master only connect to the master
    } else {
        char hostPort[50];
        sprintf(hostPort, "tcp://%s:%u", masterIp.c_str(), serverPort);

        subscriber.connect(hostPort);
    }

    subscriber.setsockopt(ZMQ_SUBSCRIBE, NULL, 0);

    if (master) {
        unsigned remaining = allNodeIps.size() - 1;

        // Keep polling until all workers reponsd
        while (remaining > 0) {
            zmq::message_t outMsg1(sizeof(unsigned));  // Send msg 1
            unsigned ctrlMsg = CTRL_MSG::MASTERUP;
            std::memcpy(outMsg1.data(), &ctrlMsg, sizeof(unsigned));
            publisher.ksend(outMsg1);

            std::this_thread::sleep_for(std::chrono::milliseconds(500));

            zmq::message_t inMsg;
            while (subscriber.krecv(&inMsg, ZMQ_DONTWAIT)) {    // wait on ALL msg 2
                unsigned inCtrlMsg;
                std::memcpy(&inCtrlMsg, inMsg.data(), inMsg.size());
                if (inCtrlMsg == CTRL_MSG::WORKERUP) {
                    --remaining;
                }
            }

        }

        zmq::message_t outMsg2(sizeof(unsigned)); // Send msg 3 (init finished)
        unsigned doneMsg = CTRL_MSG::INITDONE;
        std::memcpy(outMsg2.data(), &doneMsg, sizeof(unsigned));
        publisher.send(outMsg2);

    // Worker nodes
    } else {
        zmq::message_t inMsg;   // Recv msg 1
        while (subscriber.recv(&inMsg)) {
            unsigned msgType;
            std::memcpy(&msgType, inMsg.data(), inMsg.size());

            if (msgType == CTRL_MSG::MASTERUP) 
                break;
        }

        zmq::message_t outMsg(sizeof(unsigned));;  // Send msg 2 (ack)
        unsigned outMsgType = CTRL_MSG::WORKERUP;
        std::memcpy(outMsg.data(), &outMsgType, sizeof(unsigned));
        publisher.send(outMsg);

        workerLog("Waiting for done message");
        while (subscriber.recv(&inMsg)) {
            workerLog("Inside while");
            unsigned doneMsg;
            std::memcpy(&doneMsg, inMsg.data(), sizeof(unsigned));
            if (doneMsg == CTRL_MSG::INITDONE)
                break;
        }
        workerLog("Finished init");
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
    // Read the layer config file. Each line is a number of features
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

    // Assert there is at least one layer (input -> output)
    assert(dims.size() > 1);

    // If master node, initialize the weight matrices according to the layer config
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

//        for (unsigned u = 0; u < weightMats.size(); ++u)
//            fprintf(stdout, "Layer %u - Weights: %s\n", u, weightMats[u].shape().c_str());
    }
}

void
WeightServer::distributeWeightMatrices() {
    if (master) {
        // Master sends all the weight matrices to the worker nodes
        for (unsigned i = 0; i < weightMats.size(); ++i) {
            Matrix& weights = weightMats[i];

            zmq::message_t weightData(weights.getDataSize());
            std::memcpy((char*) weightData.data(), weights.getData(), weights.getDataSize());
            if (i == weightMats.size() - 1)
                publisher.send(weightData);
            else
                publisher.send(weightData, ZMQ_SNDMORE);
            masterLog("Sending weight matrix to socket");
        }

        // Get an ack from every worker node that they have received the weights
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

    // Worker code
    } else {
        // Worker receives each weight matrix
        int more = 0;
        do {
            zmq::message_t weightData;
            workerLog("Waiting on weightData");
            subscriber.recv(&weightData);

            char* matxData = new char[weightData.size()];
            std::memcpy(matxData, weightData.data(), weightData.size());

            workerLog("Pushing matrix onto list");
            weightMats.push_back(Matrix(dims[count], dims[count+1], matxData));
            ++count;

            size_t more_size = sizeof(more);
            subscriber.getsockopt(ZMQ_RCVMORE, &more, &more_size);
        } while (more);

        // After all matrices have been received, alert the master
        unsigned msgType = CTRL_MSG::ACK;
        zmq::message_t ackMsg(sizeof(unsigned));
        std::memcpy(ackMsg.data(), &msgType, sizeof(unsigned));
        publisher.send(ackMsg);
    }

    if (master) {
        std::cout << "All nodes up to date" << std::endl;
    }
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
