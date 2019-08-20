#include "serverworker.hpp"


extern std::mutex m, term_mutex, update_mutex;
extern std::condition_variable cv;
extern bool finished;


ServerWorker::ServerWorker(zmq::context_t& ctx_, int sock_type, unsigned& counter,
         std::vector<Matrix>& _weights, std::vector<Matrix>& _updates,
         std::vector<unsigned>& _numLambdas, WeightServer& _ws)
    : ctx(ctx_), worker(ctx, sock_type), weight_list(_weights),
    updates(_updates), numLambdas(_numLambdas), count(counter), ws(_ws) { }


// Listens on lambda threads' request for weights.
void ServerWorker::work() {
    worker.connect("inproc://backend");

    std::cout << "[Weight] Starts listening for lambdas' requests..." << std::endl;
    try {
        while (true) {
            zmq::message_t identity;
            zmq::message_t header;
            worker.recv(&identity);
            worker.recv(&header);
                
            unsigned op = parse<unsigned>((char *) header.data(), 0);
            unsigned layer = parse<unsigned>((char *) header.data(), 1);

            if (op == OP::PULL_FORWARD || op == OP::PUSH_BACKWARD) {
                std::string opStr = op == OP::PUSH_BACKWARD ? "Push" : "Pull";
                std::string accMsg = "[ACCEPTED] " + opStr + " for layer "
                                   + std::to_string(layer);
                std::cout << accMsg << std::endl;
            }

            switch (op) {
                case (OP::PULL_FORWARD):
                    sendWeights(worker, identity, layer);
                    break;
                case (OP::PUSH_BACKWARD):
                    recvUpdates(worker, identity, layer, header);
                    break;
                // Used to tell the weight server how many lambda threads
                // it should expect for this round of backprop
                case (OP::INFO):
                    updateBackpropIterationInfo(layer, header);
                    break;
                case (OP::TERM):
                    terminateServer(worker, identity);
                    break;
                default:
                    std::cerr << "ServerWorker: Unknown Op code received." << std::endl;
            }
        }
    } catch (std::exception& ex) {
        std::cerr << ex.what() << std::endl;
    }
}

void ServerWorker::sendWeights(zmq::socket_t& socket, zmq::message_t& client_id, unsigned layer) {
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

// Receive a given update from a worker
// If all updates have been received for this batch, alert the weight server
// that it is time to average and apply them
void ServerWorker::recvUpdates(zmq::socket_t& socket, zmq::message_t& client_id, unsigned layer, zmq::message_t& header) {
    zmq::message_t data;
    socket.recv(&data);

    // send ACK message back to server
    zmq::message_t confirm;
    socket.send(client_id, ZMQ_SNDMORE);
    socket.send(confirm);

    // grab lock then sum the data received with the current update matrix
    std::lock_guard<std::mutex> update_lock(update_mutex);
    cblas_saxpy(weight_list[layer].getNumElemts(), -1.0, (float*) data.data(),
                1, weight_list[layer].getData(), 1);

//    cblas_saxpy(updates[layer].getNumElemts(), 1.0,
//                (float*) data.data(), 1, updates[layer].getData(), 1);
//    ++count;
//
//    // If all the lambda results have been collected, reset the counter
//    // and tell the weight server to average and apply updates
//    if (count == numLambdas[layer]) {
//        count = 0;
//
//        std::thread t([&]{
//            ws.applyUpdates(layer);
//        });
//        t.detach();
//    }
}

// Update the weight server with the number of lambdas being called for
// this iteration so it knows when to average
void ServerWorker::updateBackpropIterationInfo(unsigned layer, zmq::message_t& header) {
    unsigned nLambdas = parse<unsigned>((char*) header.data(), 2);

    std::cout << "Number of lambdas for layer " << layer << " is "
      << nLambdas << std::endl;

    // This is not a thread-safe call but as the coordination server should
    // only send one info message per server it should be fine
    numLambdas[layer] = nLambdas;
}

// After receiving the termination message from the coordination server
// alert the main thread that it can shutdown
void ServerWorker::terminateServer(zmq::socket_t& socket, zmq::message_t& client_id) {
    std::cerr << "Server shutting down..." << std::endl;

    std::lock_guard<std::mutex> lock(m);
    finished = true;
    cv.notify_one();
}
