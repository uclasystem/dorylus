#include "weightserver.hpp"

WeightServer::WeightServer(unsigned _port, std::string& configFileName)
    : ctx(1), frontend(ctx, ZMQ_ROUTER), backend(ctx, ZMQ_DEALER), port(_port) {

    // Read in layer configurations.
    initializeWeightMatrices(configFileName);

    // Currently using randomly generated weights.
    auto seed = 8888;
    std::default_random_engine dre(seed);
    std::uniform_real_distribution<FeatType> dist(-1.5, 1.5);

    for (uint32_t u = 0; u < dims.size() - 1; ++u) {
        uint32_t dataSize = dims[u] * dims[u + 1];
        FeatType *dptr = new FeatType[dataSize];
        // Second dptr is for update matrices
        FeatType *dptr2 = new FeatType[dataSize];
        for (uint32_t ui = 0; ui < dataSize; ++ui) {
            dptr[ui] = dist(dre);
            dptr2[ui] = 0.0;
        }

        layers.push_back(Matrix(dims[u], dims[u + 1], dptr));
        updates.push_back(Matrix(dims[u], dims[u + 1], dptr2));
    }

    for (uint32_t u = 0; u < layers.size(); ++u)
        fprintf(stdout, "Layer %u Weights: %s\n", u, layers[u].shape().c_str());

    numLambdas.resize(layers.size());
    count = 0;
}


// Runs the weightserver, start a bunch of worker threads and create a proxy through frontend to
// backend.
void WeightServer::run() {
    char host_port[50];
    sprintf(host_port, "tcp://*:%u", port);
    std::cout << "Binding weight server to " << host_port << "..." << std::endl;
    frontend.bind(host_port);
    backend.bind("inproc://backend");

    std::vector<ServerWorker *> workers;
    std::vector<std::thread *> worker_threads;
    WeightServer& me = *this;
    for (int i = 0; i < kMaxThreads; ++i) {
        workers.push_back(new ServerWorker(ctx, ZMQ_DEALER, count, layers, updates, numLambdas, me));

        worker_threads.push_back(new std::thread(std::bind(&ServerWorker::work, workers[i])));
        worker_threads[i]->detach();
    }

    try {
        zmq::proxy(static_cast<void *>(frontend), static_cast<void *>(backend), nullptr);
    } catch (std::exception& ex) {
        std::cerr << "[ERROR] " << ex.what() << std::endl;
    }

    for (int i = 0; i < kMaxThreads; ++i) {
        delete worker_threads[i];
        delete workers[i];
    }
    for (Matrix& mat : layers)
        delete mat.getData();
}

void WeightServer::applyUpdates(int32_t layer) {
    std::cout << "Apply" << std::endl;
}

// Read in layer configurations.
void WeightServer::initializeWeightMatrices(std::string& configFileName) {
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
    assert(argc == 3);
    unsigned weightserverPort = std::atoi(argv[1]);
    std::string configFileName = argv[2];

    WeightServer ws(weightserverPort, configFileName);
    
    // Run in a detached thread because so that we can wait
    // on a condition variable
    std::thread t([&]{
        ws.run();
    });
    t.detach();

    // Wait for one of the threads to mark the finished bool true
    // then end the main thread
    std::unique_lock<std::mutex> lk(m);
    cv.wait(lk, [&]{ return finished; });
    
    return 0;
}
