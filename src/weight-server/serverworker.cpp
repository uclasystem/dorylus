#include "serverworker.hpp"


static void nofree(void* data, void* hint) {}

extern std::mutex term_mutex, update_mutex;
extern std::condition_variable cv;
extern bool finished;


/**
 *
 * ServerWorker constructor & destructor.
 *
 */
ServerWorker::ServerWorker(zmq::context_t& ctx_, WeightServer& _ws,
                           std::vector< TensorMap >& updateStore_,
                           std::vector< TensorMap >& _weightsStore,
                           unsigned& numLambdas_,unsigned& lambdaRecved_)
    : ctx(ctx_), workersocket(ctx, ZMQ_DEALER), ws(_ws),
      updateStore(updateStore_), weightsStore(_weightsStore),
      numLambdas(numLambdas_),lambdaRecved(lambdaRecved_) {
    workersocket.setsockopt(ZMQ_BACKLOG, 500);
    workersocket.connect("inproc://backend");
}


ServerWorker::~ServerWorker() {
    workersocket.setsockopt(ZMQ_LINGER, 0);
    workersocket.close();
}


/**
 *
 * Listen on lambda threads' requests.
 *
 */
void
ServerWorker::work() {
    std::cout << "[ Weight ] Starts listening for lambdas' requests..." << std::endl;
    try {
        while (true) {
            zmq::message_t identity;
            zmq::message_t header;
            workersocket.recv(&identity);
            workersocket.recv(&header);

            OP op = parse<OP>((char *) header.data(), 0);

            switch (op) {
                case (OP::PUSH): {
                    Chunk chunk;
                    memcpy(&chunk, (char *)header.data() + sizeof(OP), sizeof(Chunk));
                    recvTensors(identity, chunk);
                    break;
                }
                case (OP::PULL): {
                    Chunk chunk;
                    memcpy(&chunk, (char *)header.data() + sizeof(OP), sizeof(Chunk));
                    sendTensors(identity, chunk);
                    break;
                }
                case (OP::INFO): { // Used to tell how many lambda threads it should expect for this round.
                    unsigned arg = parse<unsigned>((char *) header.data(), 1);
                    setBackpropNumLambdas(identity, arg);
                    break;
                }
                case (OP::TERM): {
                    terminateServer(identity);
                    break;
                }
                default: {
                    std::cout << "Unknown op, message size: " << identity.size() << " " <<
                    header.size() << std::endl;
                    break;  /** Not an op that I care about. */
                }
            }
        }
    } catch (std::exception& ex) { /** Context Termintated. */ }
}


void ServerWorker::sendTensors(zmq::message_t& client_id, Chunk &chunk) {
    unsigned more = 1;
    workersocket.send(client_id, ZMQ_SNDMORE);
    // Weights are not yet up to date
    // if (layer == 0 && forward && !ws.servers_updates_done) {
    //     zmq::message_t header(TENSOR_HDR_SIZE);
    //     populateHeader(header.data(), ERR_HEADER_FIELD);
    //     workersocket.send(header);

    //     // clear buffer of requests before returning
    //     size_t usize = sizeof(more);
    //     while (more) {
    //         zmq::message_t tensorHeader(TENSOR_HDR_SIZE);
    //         workersocket.recv(&tensorHeader);

    //         workersocket.getsockopt(ZMQ_RCVMORE, &more, &usize);
    //     }
    // }

    TensorMap& weights = weightsStore[chunk.layer];
    while (more) {
        zmq::message_t tensorHeader(TENSOR_HDR_SIZE);
        workersocket.recv(&tensorHeader);

        std::string name = parseName((char*)tensorHeader.data());
        auto found = weights.find(name);
        if (found == weights.end()) {
            std::cerr << "Requested tensor '" << name << "' not found" << std::endl;
            zmq::message_t errorHeader(TENSOR_HDR_SIZE);
            populateHeader(errorHeader.data(), ERR_HEADER_FIELD, name.c_str());
            workersocket.send(errorHeader);
            return;
        } else {
            Matrix& reqMatrix = found->second;
            sendTensor(reqMatrix, more);
        }
    }
}


void ServerWorker::recvTensors(zmq::message_t& client_id, Chunk &chunk) {
    unsigned more = 1;
    TensorMap& weights = weightsStore[chunk.layer];
    while (more) {
        recvUpdateTensor(chunk.layer, weights);

        size_t usize = sizeof(more);
        workersocket.getsockopt(ZMQ_RCVMORE, &more, &usize);
    }
}

void ServerWorker::sendTensor(Matrix& tensor, unsigned& more) {
    zmq::message_t responseHeader(TENSOR_HDR_SIZE);
    populateHeader(responseHeader.data(), OP::PULL, tensor.name().c_str(),
      tensor.getRows(), tensor.getCols());
    unsigned bufSize = tensor.getRows() * tensor.getCols() * sizeof(FeatType);
    zmq::message_t tensorData(tensor.getData(), bufSize, nofree, NULL);

    workersocket.send(responseHeader, ZMQ_SNDMORE);

    size_t usize = sizeof(unsigned);
    workersocket.getsockopt(ZMQ_RCVMORE, &more, &usize);
    if (!more) {
        workersocket.send(tensorData);
    } else {
        workersocket.send(tensorData, ZMQ_SNDMORE);
    }
}

void ServerWorker::recvUpdateTensor(unsigned layer, TensorMap& weights) {
    zmq::message_t tensorHeader(TENSOR_HDR_SIZE);
    zmq::message_t tensorData;

    workersocket.recv(&tensorHeader);
    workersocket.recv(&tensorData);

    std::string name = parseName((char*)tensorHeader.data());
    if (weights.find(name) == weights.end()) {
        std::cerr << "Pushed tensor '" << name
          << "' not found. Make sure to allocate it before starting workers!" << std::endl;
    }

    {
        std::lock_guard<std::mutex> update_lock(update_mutex);
        lambdaRecved++;
        if (ws.servers_updates_done) {
            ws.servers_updates_done = false;
        }

        FeatType* updateSum = updateStore[layer][name].getData();
        FeatType* newUpdate = (FeatType*) tensorData.data();
        const unsigned numElemts = updateStore[layer][name].getNumElemts();
        for (unsigned u = 0; u < numElemts; ++u)
            updateSum[u] += newUpdate[u];
    }

    if (numLambdas == lambdaRecved)
        ws.applyUpdate(layer, name);
}

/**
 *
 * Update the weightserver with number of lambdas being called for this iteration.
 * Therefore it knows when to average.
 *
 */
void
ServerWorker::setBackpropNumLambdas(zmq::message_t& client_id, unsigned numLambdas_) {
    std::lock_guard<std::mutex> update_lock(update_mutex);
    numLambdas = numLambdas_;
    std::cout << "[  INFO  ] Number of lambdas set to " << numLambdas << "." << std::endl;

    // Send confirm ACK message.
    zmq::message_t confirm;
    workersocket.send(client_id, ZMQ_SNDMORE);
    workersocket.send(confirm);
}


/**
 *
 * After receiving the termination message from the graph server alert
 * the main thread that it can shutdown.
 *
 */
void
ServerWorker::terminateServer(zmq::message_t& client_id) {
    // Send confirm ACK message.
    // zmq::message_t confirm;
    // workersocket.send(client_id, ZMQ_SNDMORE);
    // workersocket.send(confirm);

    std::cerr << "[SHUTDOWN] Server shutting down..." << std::endl;

    std::lock_guard<std::mutex> lk(term_mutex);
    finished = true;
    cv.notify_one();
}
