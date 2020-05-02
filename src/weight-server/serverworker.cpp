#include "serverworker.hpp"


static void nofree(void* data, void* hint) {}

/**
 *
 * ServerWorker constructor & destructor.
 *
 */
ServerWorker::ServerWorker(zmq::context_t& ctx_, WeightServer& _ws)
    : ctx(ctx_), workersocket(ctx, ZMQ_DEALER), ws(_ws) {
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
                    setNumLambdas(identity, arg);
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
    WeightTensorMap& weights = ws.weightsStore[chunk.layer];
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
            Matrix& reqMatrix = found->second.getMat(chunk);
            sendTensor(reqMatrix, more);
        }
    }
}

void ServerWorker::recvTensors(zmq::message_t& client_id, Chunk &chunk) {
    unsigned more = 1;
    WeightTensorMap& weights = ws.weightsStore[chunk.layer];
    while (more) {
        recvUpdateTensor(chunk, weights);

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

void ServerWorker::recvUpdateTensor(Chunk &chunk, WeightTensorMap& weights) {
    zmq::message_t tensorHeader(TENSOR_HDR_SIZE);
    zmq::message_t tensorData;

    workersocket.recv(&tensorHeader);
    workersocket.recv(&tensorData);

    std::string name = parseName((char*)tensorHeader.data());
    auto found = weights.find(name);
    if (found == weights.end()) {
        std::cerr << "Pushed tensor '" << name
          << "' not found. Make sure to allocate it before starting workers!" << std::endl;
    } else {
        found->second.decRef(chunk);
        FeatType* newUpdate = (FeatType*) tensorData.data();
        unsigned localUpdCnt = ws.weightsStore[chunk.layer][name].localUpdate(newUpdate);

        if (ws.numLambdas == localUpdCnt) {
            ws.applyUpdate(chunk.layer, name);
        }
    }
}

/**
 *
 * Update the weightserver with number of lambdas being called for this iteration.
 * Therefore it knows when to average.
 *
 */
void
ServerWorker::setNumLambdas(zmq::message_t& client_id, unsigned numLambdas) {
    // Send confirm ACK message.
    zmq::message_t confirm;
    workersocket.send(client_id, ZMQ_SNDMORE);
    workersocket.send(confirm);

    ws.setLocalUpdTot(numLambdas);
    std::cout << "[  INFO  ] Number of lambdas set to " << numLambdas << "." << std::endl;
}


/**
 *
 * After receiving the termination message from the graph server alert
 * the main thread that it can shutdown.
 *
 */
void
ServerWorker::terminateServer(zmq::message_t& client_id) {
    std::cerr << "[SHUTDOWN] Server shutting down..." << std::endl;

    std::lock_guard<std::mutex> lk(ws.termMtx);
    ws.term = true;
    ws.termCV.notify_one();
}
