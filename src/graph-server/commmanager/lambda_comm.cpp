#include "lambda_comm.hpp"


std::mutex m;
std::condition_variable cv;
std::mutex ServerWorker::count_mutex;


/**
 *
 * ServerWorker is a wrapper over the sender & receiver thread.
 * 
 */
void
ServerWorker::work() {
	worker.connect("inproc://backend");

	try {
		while (true) {
			zmq::message_t identity;
			zmq::message_t header;
			worker.recv(&identity);
			worker.recv(&header);

			int32_t chunkId = parse<int32_t>((char *) identity.data(), 0);
			int32_t op = parse<int32_t>((char *) header.data(), 0);
			int32_t partId = parse<int32_t>((char *) header.data(), 1);
			int32_t rows = parse<int32_t>((char *) header.data(), 2);
			int32_t cols = parse<int32_t>((char *) header.data(), 3);

			std::string opStr = op == 0 ? "Push" : "Pull";
			std::string accMsg = opStr + " from thread " + std::to_string(chunkId)
			                   + " for partition " + std::to_string(partId);
			printLog(nodeId, "AccMSG: %s.\n", accMsg.c_str());

			switch (op) {
				case (OP::PULL):
					sendMatrixChunk(worker, identity, partId);
					break;
				case (OP::PUSH):
					recvMatrixChunks(worker, partId, rows, cols);
					break;
				default:
					printLog(nodeId, "ServerWorker: Unknown Op code received.\n");
			}
		}
	} catch (std::exception& ex) {
		std::cerr << ex.what() << std::endl;
	}
}


void
ServerWorker::sendMatrixChunk(zmq::socket_t& socket, zmq::message_t& client_id, int32_t partId) {
	zmq::message_t header(HEADER_SIZE);

	// Reject a send request if the partition id is invalid.
	if (partId >= nParts) {
		populateHeader((char *) header.data(), -1, -1, -1, -1);
		socket.send(client_id);
		socket.send(header);

	// Partition id is valid, so send the matrix segment.
	} else {

		// Check to make sure that the bounds of this partition do not exceed the bounds of the data array.
		// If they do, set partition end to the end of the array.
		int32_t diff = 0;
		int32_t thisPartRows = partRows;
		int32_t thisBufSize = bufSize;
		if ((partId * partRows + partRows) > matrix.rows) {
			diff = (partId * partRows + partRows) - matrix.rows;
			thisPartRows = partRows - diff;
			thisBufSize = thisPartRows * partCols * sizeof(FeatType);
		}

		populateHeader((char *) header.data(), OP::RESP, 0, thisPartRows, partCols);
		FeatType *partition_start = (matrix.getData()) + (partId * offset);
		zmq::message_t partitionData(thisBufSize);
		std::memcpy(partitionData.data(), partition_start, thisBufSize);

		socket.send(client_id, ZMQ_SNDMORE);
		socket.send(header, ZMQ_SNDMORE);
		socket.send(partitionData);
	}
}

void
ServerWorker::recvMatrixChunks(zmq::socket_t& socket, int32_t partId, int32_t rows, int32_t cols) {
	uint32_t offset = partId * partRows * cols;
	FeatType *thisPartitionZStart = zData + offset;
	FeatType *thisPartitionActStart = actData + offset;

	zmq::message_t data;
	socket.recv(&data);
	std::memcpy(thisPartitionZStart, data.data(), data.size());

	socket.recv(&data);
	std::memcpy(thisPartitionActStart, data.data(), data.size());

	Matrix z(rows, cols, thisPartitionZStart);
	Matrix act(rows, cols, thisPartitionActStart);

	std::lock_guard<std::mutex> lock(count_mutex);
	++count;

	if (count == nParts)
		cv.notify_one();
}


/**
 *
 * Refresh the member values.
 * 
 */
void
ServerWorker::refreshState(FeatType *zData_, FeatType *actData_, int32_t nextIterCols_) {
	nextIterCols = nextIterCols_;
	zData = zData_;
	actData = actData_;

	partCols = matrix.cols;
	partRows = std::ceil((float) matrix.rows / (float) nParts);
	offset = partRows * partCols;
	bufSize = offset * sizeof(FeatType);
}


/**
 *
 * Call startContext() before the lambda invokation to refresh the parameters, and call endContext() after
 * the global barrier to revoke unused memory space.
 * 
 */
void
LambdaComm::startContext(FeatType *dataBuf_, int32_t rows_, int32_t cols_, int32_t nextIterCols_, unsigned layer_) {
	nextIterCols = nextIterCols_;
	counter = 0;
	layer = layer_;
	matrix = Matrix(rows_, cols_, dataBuf_);
	zData = new FeatType[rows_ * nextIterCols_];
	actData = new FeatType[rows_ * nextIterCols_];
	printLog(nodeId, "New lambda communication context created on layer %u.\n", layer);

	for (auto&& worker : workers)
		worker->refreshState(zData, actData, nextIterCols);
}

void
LambdaComm::endContext() {
	counter = 0;
    delete[] zData;
    delete[] matrix.data;   // Delete last iter's data buffer. The engine must reset its buf ptr to getActivationData().
    printLog(nodeId, "Lambda communication context on layer %u finished.\n", layer);
}


/**
 *
 * When a lambda connection is desired.
 * 
 */
void
LambdaComm::requestLambdas() {

	printLog(nodeId, "Sending lambda threads requests to coordserver...\n");

	zmq::message_t header(HEADER_SIZE);
	populateHeader((char *) header.data(), OP::REQ, layer, nParts);
	sendsocket.send(header, ZMQ_SNDMORE);

	zmq::message_t ip_msg(nodeIp.size());
	std::memcpy(ip_msg.data(), nodeIp.c_str(), nodeIp.size());
	sendsocket.send(ip_msg);
	
	zmq::message_t reply;
	sendsocket.recv(&reply);

	printLog(nodeId, "Coordserver accepts the request. Waiting on results...\n");

	// Block until all parts have been handled.
	std::unique_lock<std::mutex> lk(m);
	cv.wait(lk, [&]{ return counter == nParts; });
}
