#include "lambda_comm.hpp"


std::mutex m;
std::condition_variable cv;


server_worker::server_worker(zmq::context_t& ctx_, int sock_type,
  int32_t nParts_, int32_t& counter_, std::shared_ptr<Matrix> data_) 
  : ctx(ctx_),
  worker(ctx, sock_type),
  nParts(nParts_),
  count(counter_),
  data(data_) {
	partCols = data->cols;
	partRows = std::ceil((float)data->rows / (float)nParts);
	offset = partRows * partCols;
	bufSize = offset * sizeof(DTYPE);
}

void server_worker::work() {
	worker.connect("inproc://backend");

	try {
		while (true) {
			zmq::message_t identity;
			zmq::message_t header;
			worker.recv(&identity);
			worker.recv(&header);

			int32_t cli_id = parse<int32_t>((char*)identity.data(), 0);

			int32_t op = parse<int32_t>((char*)header.data(), 0);
			int32_t partId = parse<int32_t>((char*)header.data(), 1);
			int32_t rows = parse<int32_t>((char*)header.data(), 2);
			int32_t cols = parse<int32_t>((char*)header.data(), 3);
			int32_t resType = parse<int32_t>((char*)header.data(), 4);

			std::string opStr = op == 0 ? "Push" : "Pull";
			std::string accMsg = "[ACCEPTED] " + opStr + " from " + std::to_string(cli_id)
				+ " for partition " + std::to_string(partId);
			std::cout << accMsg << std::endl;

			switch(op) {
				case (OP::PULL):
					sendMatrixChunk(worker, identity, partId);
					break;
				case (OP::PUSH):
					recvMatrixChunks(worker, partId, rows, cols);
					break;
				default:
					std::cerr << "Unknown Op code" << std::endl;
			}
		}
	} catch (std::exception& ex) {
		std::cerr << ex.what() << std::endl;
	}
}

void server_worker::sendMatrixChunk(zmq::socket_t& socket,
  zmq::message_t& client_id, int32_t partId) {
	zmq::message_t header(HEADER_SIZE);
	if (partId >= nParts) {
		populateHeader((char*)header.data(), -1, -1, -1, -1);
		socket.send(client_id);
		socket.send(header);
	} else {
		// Check to make sure that the bounds of this partition do not exceed
		// exceed the bounds of the data array
		// If they do, set partition end to the end of the array
		int32_t diff = 0;
		int32_t thisPartRows = partRows;
		int32_t thisBufSize = bufSize;
		if ((partId * partRows + partRows) > data->rows) {
			diff = (partId * partRows + partRows) - data->rows;
			thisPartRows = partRows - diff;
			thisBufSize = thisPartRows * partCols * sizeof(DTYPE);
		}

		populateHeader((char*)header.data(), OP::RESP, 0, thisPartRows, partCols);
		DTYPE* partition_start = (data->getData()) + (partId * offset);
		zmq::message_t partitionData(thisBufSize);
		std::memcpy(partitionData.data(), partition_start, thisBufSize);

		socket.send(client_id, ZMQ_SNDMORE);
		socket.send(header, ZMQ_SNDMORE);
		socket.send(partitionData);
	}
}

void server_worker::recvMatrixChunks(zmq::socket_t& socket, int32_t partId, int32_t rows,
  int32_t cols) {
	zmq::message_t data;
	socket.recv(&data);
	char* zBuffer = new char[data.size()];
	std::memcpy(zBuffer, data.data(), data.size());

	socket.recv(&data);
	char* actBuffer = new char [data.size()];
	std::memcpy(actBuffer, data.data(), data.size());

	Matrix z(rows, cols, zBuffer);
	Matrix act(rows, cols, actBuffer);

	std::lock_guard<std::mutex> lock(count_mutex);
	++count;

	if (count == nParts) cv.notify_one();
}

std::mutex server_worker::count_mutex;

LambdaComm::LambdaComm(std::shared_ptr<Matrix> data_, unsigned port_,
  int32_t rows_, int32_t cols_, int32_t nParts_, int32_t numListeners_)
  : ctx(1),
    frontend(ctx, ZMQ_ROUTER),
    backend(ctx, ZMQ_DEALER),
    port(port_),
    data(data_),
    nParts(nParts_),
    numListeners(numListeners_),
    counter(0) {
    std::cout << "Data Matrix dimensions " << data->shape() << std::endl;
}

void LambdaComm::run() {
	char host_port[50];
	sprintf(host_port, "tcp://*:%u", port);
	frontend.bind(host_port);
	backend.bind("inproc://backend");

	std::vector<server_worker*> workers;
	std::vector<std::thread*> worker_threads;
	for (int i = 0; i < numListeners; ++i) {
		workers.push_back(new server_worker(ctx, ZMQ_DEALER, nParts, counter, data));

		worker_threads.push_back(new std::thread(std::bind(&server_worker::work, workers[i])));
		worker_threads[i]->detach();
	}

	try {
		zmq::proxy(frontend, backend, nullptr);
	} catch (std::exception& ex) {
		std::cerr << ex.what() << std::endl;
	}

	for (int i = 0; i < numListeners; ++i) {
		delete workers[i];
		delete worker_threads[i];
	}
}

void LambdaComm::requestLambdas(std::string& coordserverIp, std::string& port,
  int32_t layer) {
	char chost_port[50];
	sprintf(chost_port, "tcp://%s:%s", coordserverIp.c_str(), port.c_str());
	zmq::socket_t socket(ctx, ZMQ_PAIR);
	socket.connect(chost_port);

	zmq::message_t header(HEADER_SIZE);
	populateHeader((char*)header.data(), OP::REQ, layer, nParts);
	socket.send(header);

	std::unique_lock<std::mutex> lk(m);
	cv.wait(lk, [&]{ return counter == nParts; });

	std::cout << "All partitions received" << std::endl;
}

