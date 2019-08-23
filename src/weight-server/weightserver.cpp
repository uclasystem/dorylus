#include "weightserver.hpp"


std::mutex term_mutex, update_mutex;
std::condition_variable cv;
bool finished = false;


static std::vector<ServerWorker *> workers;
static std::vector<std::thread *> worker_threads;
static std::ofstream outfile;


#define NUM_LISTENERS 5


/**
 *
 * Weightserver constructor & destructor.
 * 
 */
WeightServer::WeightServer(unsigned _port, std::string& configFileName)
    : ctx(1), frontend(ctx, ZMQ_ROUTER), backend(ctx, ZMQ_DEALER), port(_port),
      numLambdas(20), count(0) {

    // Read in layer configurations.
    initializeWeightMatrices(configFileName);

    // TODO: Currently using randomly generated weights.
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

WeightServer::~WeightServer() {

    // Delete allocated resources.
    for (int i = 0; i < NUM_LISTENERS; ++i) {
        delete workers[i];
        delete worker_threads[i];
    }
    for (Matrix& mat : weightMats)
        delete[] mat.getData();
}


/**
 *
 * Runs the weightserver, start a bunch of worker threads and create a proxy through frontend to backend.
 * 
 */
void
WeightServer::run() {
    char host_port[50];
    sprintf(host_port, "tcp://*:%u", port);
    std::cout << "Binding weight server to " << host_port << "..." << std::endl;
    frontend.bind(host_port);
    backend.bind("inproc://backend");

    std::vector<ServerWorker *> workers;
    std::vector<std::thread *> worker_threads;
    WeightServer& me = *this;
    for (int i = 0; i < NUM_LISTENERS; ++i) {
        workers.push_back(new ServerWorker(ctx, count, me, weightMats, updates, numLambdas));
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
void WeightServer::applyUpdates() {
    std::lock_guard<std::mutex> update_lock(update_mutex);

    // Use pop() to avoid extra clearing.
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
    assert(argc == 4);
    unsigned weightserverPort = std::atoi(argv[1]);
    std::string configFileName = argv[2];

    // Set output file location.
    std::string tmpfileName = std::string(argv[3]) + "/output";
    outfile.open(tmpfileName, std::fstream::out);
    assert(outfile.good());

    WeightServer ws(weightserverPort, configFileName);
    
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
