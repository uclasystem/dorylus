#include <algorithm>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <fstream>
#include <functional>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <zmq.hpp>
#include <boost/algorithm/string/trim.hpp>
#include "../utils/utils.hpp"


std::mutex m, term_mutex;
std::condition_variable cv;
bool finished = false;

std::vector<Timer> timers;
std::mutex timerMutex;

/**
 *
 * Wrapper over a server worker thread.
 * 
 */
class ServerWorker {

public:

    ServerWorker(zmq::context_t& ctx_, int sock_type, std::vector<Matrix>& _weights)
        : ctx(ctx_), worker(ctx, sock_type), weight_list(_weights) { }

    // Listens on lambda threads' request for weights.
    void work() {
        worker.connect("inproc://backend");

        std::cout << "[Weight] Starts listening for lambdas' requests..." << std::endl;
        try {
            while (true) {
                zmq::message_t identity;
                zmq::message_t header;
                worker.recv(&identity);
                worker.recv(&header);
                    
                timerMutex.lock();
                timers.push_back(Timer());
                unsigned timerId = timers.size() - 1;
                timerMutex.unlock();
                timers[timerId].start();
                
                unsigned chunkId = parse<unsigned>((char *) identity.data(), 0);
                unsigned op = parse<unsigned>((char *) header.data(), 0);
                unsigned layer = parse<unsigned>((char *) header.data(), 1);

                if (op != OP::TERM) {
                    std::string opStr = op == 0 ? "Push" : "Pull";
                    std::string accMsg = "[ACCEPTED] " + opStr + " from thread "
                                       + std::to_string(chunkId) + " for layer "
                                       + std::to_string(layer);
                    std::cout << accMsg << std::endl;
                }

                switch (op) {
                    case (OP::PULL):
                        sendWeights(worker, identity, layer);
                        break;
                    case (OP::PUSH):
                        recvUpdates(identity, layer, header);
                        break;
                    case (OP::TERM):
                        terminateServer(worker, identity);
                        break;
                    default:
                        std::cerr << "ServerWorker: Unknown Op code received." << std::endl;
                }
                timers[timerId].stop();
            }
        } catch (std::exception& ex) {
            std::cerr << ex.what() << std::endl;
        }
    }

private:

    void sendWeights(zmq::socket_t& socket, zmq::message_t& client_id, unsigned layer) {
        Matrix& weights = weight_list[layer];
        zmq::message_t header(HEADER_SIZE);
        populateHeader((char *) header.data(), OP::RESP, 0, weights.getRows(), weights.getCols());
        
        zmq::message_t weightData(weights.getDataSize());
        std::memcpy((char *) weightData.data(), weights.getData(), weights.getDataSize());
        
        // The identity message will be implicitly consumed to route the message to the correct client.
        socket.send(client_id, ZMQ_SNDMORE);
        socket.send(header, ZMQ_SNDMORE);
        socket.send(weightData);
    }

    void recvUpdates(zmq::message_t& client_id, unsigned layer, zmq::message_t& header) {
        // TODO: Receive updates from threads.
    }

    void terminateServer(zmq::socket_t& socket, zmq::message_t& client_id) {
        std::cerr << "Server shutting down..." << std::endl;

        // Printing all the timers for each request minus the last one since that is the
        // kill message
        std::cout << "Send times: ";
        for (unsigned ui = 0; ui < timers.size()-1; ++ui) {
            std::cout << timers[ui].getTime() << " ";
        }
        std::cout << std::endl;

        std::lock_guard<std::mutex> lock(m);
        finished = true;
        cv.notify_one();
    }

    zmq::context_t &ctx;
    zmq::socket_t worker;
    std::vector<Matrix>& weight_list;
};


/**
 *
 * Class of the weightserver. Weightservers are only responsible for replying weight requests from lambdas,
 * and possibly handle weight updates.
 * 
 */
class WeightServer {

public:
    
    WeightServer(unsigned _port, std::string& configFileName)
        : ctx(1), frontend(ctx, ZMQ_ROUTER), backend(ctx, ZMQ_DEALER), port(_port) {

        // Read in layer configurations.
        initializeWeightMatrices(configFileName);

        // Currently using randomly generated weights.
        auto seed = 8888;
        std::default_random_engine dre(seed);
        std::uniform_real_distribution<FeatType> dist(-1.5, 1.5);

        for (unsigned u = 0; u < dims.size() - 1; ++u) {
            unsigned dataSize = dims[u] * dims[u + 1];
            FeatType *dptr = new FeatType[dataSize];
            for (unsigned ui = 0; ui < dataSize; ++ui)
                dptr[ui] = dist(dre);

            layers.push_back(Matrix(dims[u], dims[u + 1], dptr));
        }

        for (unsigned u = 0; u < layers.size(); ++u)
            fprintf(stdout, "Layer %u Weights: %s\n", u, layers[u].shape().c_str());
    }

    // Defines how many concurrent weightserver threads to use.
    enum { kMaxThreads = 2 };

    // Runs the weightserver, start a bunch of worker threads and create a proxy through frontend to
    // backend.
    void run() {
        char host_port[50];
        sprintf(host_port, "tcp://*:%u", port);
        std::cout << "Binding weight server to " << host_port << "..." << std::endl;
        frontend.bind(host_port);
        backend.bind("inproc://backend");

        std::vector<ServerWorker *> workers;
        std::vector<std::thread *> worker_threads;
        for (int i = 0; i < kMaxThreads; ++i) {
            workers.push_back(new ServerWorker(ctx, ZMQ_DEALER, layers));

            worker_threads.push_back(new std::thread(std::bind(&ServerWorker::work, workers[i])));
            worker_threads[i]->detach();
        }

        try {
            zmq::proxy(static_cast<void *>(frontend), static_cast<void *>(backend), nullptr);
        } catch (std::exception& ex) {
            std::cerr << "[ERROR] " << ex.what() << std::endl;
        }

        // Clean up work happens after proxy brokes.
        for (int i = 0; i < kMaxThreads; ++i) {
            delete worker_threads[i];
            delete workers[i];
        }
        for (Matrix& mat : layers)
            delete mat.getData();
    }

private:

    // Read in layer configurations.
    void initializeWeightMatrices(std::string& configFileName) {
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

    std::vector<unsigned> dims;
    std::vector<Matrix> layers;

    zmq::context_t ctx;
    zmq::socket_t frontend;
    zmq::socket_t backend;
    unsigned port;
};


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
