#include "serverworker.hpp"


extern std::mutex term_mutex, update_mutex;
extern std::condition_variable cv;
extern bool finished;


/**
 *
 * ServerWorker constructor & destructor.
 *
 */
ServerWorker::ServerWorker(zmq::context_t& ctx_, unsigned& counter, WeightServer& _ws,
                           std::vector<Matrix>& weights_, std::vector<Matrix>& updates_, unsigned& numLambdas_)
    : ctx(ctx_), workersocket(ctx, ZMQ_DEALER), count(counter), ws(_ws),
      weightMats(weights_), updateMats(updates_), numLambdas(numLambdas_) {
    workersocket.setsockopt(ZMQ_LINGER, 0);
    workersocket.connect("inproc://backend");
}


ServerWorker::~ServerWorker() {
    std::cout << "Closing sockets" << std::endl;
    workersocket.close();

    std::cout << "Closing context" << std::endl;
    ctx.close();
}


/**
 *
 * Listen on lambda threads' requests.
 *
 */
void
ServerWorker::work() {
    std::cout << "[Weight] Starts listening for lambdas' requests..." << std::endl;
    try {
        while (true) {
            zmq::message_t identity;
            zmq::message_t header;
            workersocket.recv(&identity);
            workersocket.recv(&header);

            unsigned op = parse<unsigned>((char *) header.data(), 0);
            unsigned arg = parse<unsigned>((char *) header.data(), 1);

            std::string accMsg;
            if (op == OP::PULL_FORWARD)
                accMsg = "[ACCEPTED] Pull FORWARD for layer " + std::to_string(arg) + ".";
            else if (op == OP::PULL_BACKWARD)
                accMsg = "[ACCEPTED] Pull BACKWARD from layer " + std::to_string(arg) + ".";
            else if (op == OP::PUSH_BACKWARD)
                accMsg = "[ UPDATE ] Push BACKWARD from layer " + std::to_string(arg) + ".";
            if (!accMsg.empty())
                std::cout << accMsg << std::endl;

            switch (op) {
                case (OP::PULL_FORWARD):
                    sendWeights(identity, arg);
                    break;
                case (OP::PULL_BACKWARD):
                    sendWeights(identity, arg);
                    break;
                case (OP::PUSH_BACKWARD):
                    recvUpdate(identity, arg);
                    break;
                case (OP::INFO):    // Used to tell how many lambda threads it should expect for this round.
                    setBackpropNumLambdas(identity, arg);
                    break;
                case (OP::TERM):
                    terminateServer(identity);
                    break;
                default:
                    std::cout << "Unknown op" << std::endl;
                    break;  /** Not an op that I care about. */
            }
        }
    } catch (std::exception& ex) { /** Context Termintated. */ }
}


/**
 *
 * Send weight matrix to lambdas.
 *
 */
void
ServerWorker::sendWeights(zmq::message_t& client_id, unsigned layer) {
    if (layer >= weightMats.size()) {
        std::cerr << "[ERROR] No such weights corresponding to layer " << layer << std::endl;
    }

    workersocket.send(client_id, ZMQ_SNDMORE);    // The identity message will be implicitly consumed to route to the correct client.

    zmq::message_t header(HEADER_SIZE);
    populateHeader((char *) header.data(), OP::RESP, 0, weightMats[layer].getRows(), weightMats[layer].getCols());
    workersocket.send(header, ZMQ_SNDMORE);

    zmq::message_t weightData(weightMats[layer].getDataSize());
    std::memcpy((char *) weightData.data(), weightMats[layer].getData(), weightMats[layer].getDataSize());
    workersocket.send(weightData);
}


/**
 *
 * Send weight matrices 2 -> last to lambdas for backward-prop computation.
 *
 */
void
ServerWorker::sendWeightsBackward(zmq::message_t& client_id) {
    workersocket.send(client_id, ZMQ_SNDMORE);

    for (unsigned i = 1; i < weightMats.size(); ++i) {
        Matrix& weightMat = weightMats[i];

        zmq::message_t header(HEADER_SIZE);
        populateHeader((char *) header.data(), OP::RESP, 0, weightMat.getRows(), weightMat.getCols());
        workersocket.send(header, ZMQ_SNDMORE);

        zmq::message_t weightData(weightMat.getDataSize());
        std::memcpy((char *) weightData.data(), weightMat.getData(), weightMat.getDataSize());
        if (i == weightMats.size() - 1)
            workersocket.send(weightData);
        else
            workersocket.send(weightData, ZMQ_SNDMORE);
    }
}

/**
 *
 * Receive a given update from a worker. If all udpates have been received, alert the weight server that it is
 * time to average and apply them.
 *
 */
void
ServerWorker::recvUpdate(zmq::message_t& client_id, unsigned layer) {
    zmq::message_t updateMsg;
    workersocket.recv(&updateMsg);

    // Send confirm ACK message.
    zmq::message_t confirm;
    workersocket.send(client_id, ZMQ_SNDMORE);
    workersocket.send(confirm);

    float *updateSum = updateMats[layer].getData();
    float *updateNew = (float *) updateMsg.data();

    // Grab lock then sum the data received in this update matrix.
    std::lock_guard<std::mutex> update_lock(update_mutex);
    for (unsigned u = 0; u < updateMats[layer].getNumElemts(); ++u)
        updateSum[u] += updateNew[u];

    // if (layer == 0) {
    numLambdas--;

    // If this is the final update, begin global aggregation.
    if (numLambdas == 0) {
        ws.applyUpdate(layer);
        // ws.applyUpdates();
    }
    // }
}

/**
 *
 * Receive all weight updates from a worker. If all udpates have been received, alert the weight server that it is
 * time to average and apply them.
 *
 */
void
ServerWorker::recvUpdates(zmq::message_t& client_id) {
    for (unsigned i = 0; i < updateMats.size(); ++i) {
        zmq::message_t updateMsg;
        workersocket.recv(&updateMsg);

        float *updateSum = updateMats[i].getData();
        float *updateNew = (float *) updateMsg.data();

        // Grab lock then sum the data received in this update matrix.
        std::lock_guard<std::mutex> update_lock(update_mutex);
        for (unsigned u = 0; u < updateMats[i].getNumElemts(); ++u)
            updateSum[u] += updateNew[u];

        // If I have received all updates from this Lambda decrement the counter.
        if (i == updateMats.size() - 1) {
            numLambdas--;

            // If this is the final update, begin global aggregation.
            if (numLambdas == 0)
                ws.applyUpdates();
        }
    }

    // Send confirm ACK message.
    zmq::message_t confirm;
    workersocket.send(client_id, ZMQ_SNDMORE);
    workersocket.send(confirm);
}


/**
 *
 * Update the weightserver with number of lambdas being called for this iteration.
 * Therefore it knows when to average.
 *
 */
void
ServerWorker::setBackpropNumLambdas(zmq::message_t& client_id, unsigned numLambdas_) {

    // This is not a thread-safe call, but as the coordination server should
    // only send one info message per server, it should be fine.
    std::lock_guard<std::mutex> update_lock(update_mutex);
    numLambdas += numLambdas_;
    std::cout << "[  INFO  ] Number of lambdas set to " << numLambdas << "." << std::endl;

    // Send confirm ACK message.
    zmq::message_t confirm;
    workersocket.send(client_id, ZMQ_SNDMORE);
    workersocket.send(confirm);
}


/**
 *
 * After receiving the termination message from the coordination server alert
 * the main thread that it can shutdown.
 *
 */
void
ServerWorker::terminateServer(zmq::message_t& client_id) {

    // Send confirm ACK message.
    zmq::message_t confirm;
    workersocket.send(client_id, ZMQ_SNDMORE);
    workersocket.send(confirm);

    std::cerr << "[SHUTDOWN] Server shutting down..." << std::endl;

    std::lock_guard<std::mutex> lk(term_mutex);
    finished = true;
    cv.notify_one();
}
