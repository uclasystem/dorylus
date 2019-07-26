#include "lambda_comm.hpp"

std::mutex m;
std::condition_variable cv;

void populateHeader(char* header, int32_t op, int32_t id, int32_t rows,
    int32_t cols) {
    serialize<int32_t>(header, 0, op);
    serialize<int32_t>(header, 1, id);
    serialize<int32_t>(header, 2, rows);
    serialize<int32_t>(header, 3, cols);
}


server_worker::server_worker(zmq::context_t& ctx_, int sock_type,
  int32_t nParts_, int32_t nextIterCols_, int32_t& counter_, Matrix& data_,
  FeatType* zData_, FeatType* actData_) 
  : data(data_),
    nextIterCols(nextIterCols_),
    zData(zData_),
    actData(actData_),
    ctx(ctx_),
    worker(ctx, sock_type),
    nParts(nParts_),
    count(counter_) {
	partCols = data.cols;
	partRows = std::ceil((float)data.rows / (float)nParts);
	offset = partRows * partCols;
	bufSize = offset * sizeof(FeatType);

	fprintf(stderr, "worker data.shape(): %s\n", data.shape().c_str());
	
	fprintf(stderr, "partRows: %d, partCols: %d, offset: %d\n", partRows,
	  partCols, offset);
}

void server_worker::work() {
	worker.connect("inproc://backend");

	try {
		while (true) {
			fprintf(stderr, "Waiting for lambda connections\n");
			zmq::message_t identity;
			zmq::message_t header;
			worker.recv(&identity);
			worker.recv(&header);

			int32_t cli_id = parse<int32_t>((char*)identity.data(), 0);

			int32_t op = parse<int32_t>((char*)header.data(), 0);
			int32_t partId = parse<int32_t>((char*)header.data(), 1);
			int32_t rows = parse<int32_t>((char*)header.data(), 2);
			int32_t cols = parse<int32_t>((char*)header.data(), 3);

			std::string opStr = op == 0 ? "Push" : "Pull";
			std::string accMsg = "[ACCEPTED] " + opStr + " from thread "
				+ std::to_string(cli_id) + " for partition "
				+ std::to_string(partId);
			std::cerr << accMsg << std::endl;

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
		fprintf(stderr, "Rejecting request for %d as this partition does not exist\n", partId);
		populateHeader((char*)header.data(), -1, -1, -1, -1);
		socket.send(client_id);
		socket.send(header);
	} else {
		fprintf(stderr, "Request for partition %d accepted\n", partId);
		// Check to make sure that the bounds of this partition do not exceed
		// exceed the bounds of the data array
		// If they do, set partition end to the end of the array
		int32_t diff = 0;
		int32_t thisPartRows = partRows;
		int32_t thisBufSize = bufSize;
		if ((partId * partRows + partRows) > data.rows) {
			diff = (partId * partRows + partRows) - data.rows;
			thisPartRows = partRows - diff;
			thisBufSize = thisPartRows * partCols * sizeof(FeatType);
		}

		fprintf(stderr, "thisPartRows: %d, partCols: %d\n", thisPartRows, partCols);

		populateHeader((char*)header.data(), OP::RESP, 0, thisPartRows, partCols);
		FeatType* partition_start = (data.getData()) + (partId * offset);
		zmq::message_t partitionData(thisBufSize);
		std::memcpy(partitionData.data(), partition_start, thisBufSize);

		socket.send(client_id, ZMQ_SNDMORE);
		socket.send(header, ZMQ_SNDMORE);
		socket.send(partitionData);
	}
}

void server_worker::recvMatrixChunks(zmq::socket_t& socket, int32_t partId,
  int32_t rows, int32_t cols) {
	uint32_t offset = partId * partRows * cols;
	FeatType* thisPartitionZStart = zData + offset;
	FeatType* thisPartitionActStart = actData + offset;

	zmq::message_t data;
	socket.recv(&data);
	std::memcpy(thisPartitionZStart, data.data(), data.size());

	socket.recv(&data);
	std::memcpy(thisPartitionActStart, data.data(), data.size());

	Matrix z(rows, cols, thisPartitionZStart);
	Matrix act(rows, cols, thisPartitionActStart);

	fprintf(stderr, "Partition %d activations\n%s\n", partId,
	  act.str().c_str());

	std::lock_guard<std::mutex> lock(count_mutex);
	++count;

	if (count == nParts) cv.notify_one();
}

std::mutex server_worker::count_mutex;

LambdaComm::LambdaComm(FeatType* data_, std::string& nodeIp_, unsigned port_,
  int32_t rows_, int32_t cols_, int32_t nextIterCols_, int32_t nParts_,
  int32_t numListeners_)
  : nextIterCols(nextIterCols_),
    nParts(nParts_),
    numListeners(numListeners_),
    counter(0),
    ctx(1),
    frontend(ctx, ZMQ_ROUTER),
    backend(ctx, ZMQ_DEALER),
    nodeIp(nodeIp_),
    port(port_) {
	data = Matrix(rows_, cols_, data_);
	std::cerr << "Data Matrix dimensions " << data.shape() << std::endl;

	zData = new FeatType[rows_ * nextIterCols];
	actData = new FeatType[rows_ * nextIterCols];
}

void LambdaComm::run() {
	char host_port[50];
	sprintf(host_port, "tcp://*:%u", port);
	std::cerr << "Binding to " << host_port << std::endl;
	frontend.bind(host_port);
	backend.bind("inproc://backend");

	fprintf(stderr, "data.shape(): %s\n", data.shape().c_str());

	std::vector<server_worker*> workers;
	std::vector<std::thread*> worker_threads;
	for (int i = 0; i < numListeners; ++i) {
		workers.push_back(new server_worker(ctx, ZMQ_DEALER, nParts,
		  nextIterCols, counter, data, zData, actData));

		worker_threads.push_back(new std::thread(std::bind(&server_worker::work, workers[i])));
		worker_threads[i]->detach();
	}

	// TODO:
	//   Either find a termination condition for this listener or 
	//   make the class a member of Engine so that it is not spawning
	//   new listeners every iteration that do not die.
	//   Probably best to integrate the listener into Engine and add 
	//   "set" APIs to change its config every iteration
	try {
		zmq::proxy(static_cast<void*>(frontend), static_cast<void*>(backend), nullptr);
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
	zmq::socket_t socket(ctx, ZMQ_REQ);
	std::cerr << "Connecting to " << chost_port << std::endl;
	socket.connect(chost_port);

	zmq::message_t header(HEADER_SIZE);
	populateHeader((char*)header.data(), OP::REQ, layer, nParts);
	socket.send(header, ZMQ_SNDMORE);

	zmq::message_t ip_msg(nodeIp.size());
	std::memcpy(ip_msg.data(), nodeIp.c_str(), nodeIp.size());
	socket.send(ip_msg);
	
	zmq::message_t reply;
	socket.recv(&reply);
	std::cerr << "Received reply. Waiting for threads to finish" << std::endl;

	std::unique_lock<std::mutex> lk(m);
	cv.wait(lk, [&]{ return counter == nParts; });

	std::cerr << "All partitions received" << std::endl;
}

FeatType* LambdaComm::getZData() { return zData; }
FeatType* LambdaComm::getActivationData() { return actData; }

