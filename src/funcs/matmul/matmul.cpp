#include <chrono>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>

// BOOST includes
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

#include <cblas.h>
#include <zmq.hpp>

// AWS cpp runtime include
#include <aws/lambda-runtime/runtime.h>

#include "../../utils/utils.h"

#define SND_MORE true
#define NO_MORE false

using boost::property_tree::ptree;
using boost::property_tree::read_json;
using boost::property_tree::write_json;
using namespace aws::lambda_runtime;
using namespace std::chrono;



Matrix requestMatrix(zmq::socket_t& socket, int32_t id) {
	zmq::message_t header(HEADER_SIZE);
	populateHeader((char*)header.data(), OP::PULL, id);
	socket.send(header);

	zmq::message_t respHeader;
	socket.recv(&respHeader);

	int32_t layerResp = parse<int32_t>((char*)respHeader.data(), 1);
	if (layerResp == -1) {
		std::cout << "No corresponding feature chunk" << std::endl;

		return Matrix();
	} else {
		int32_t rows = parse<int32_t>((char*)respHeader.data(), 2);
		int32_t cols = parse<int32_t>((char*)respHeader.data(), 3);
		zmq::message_t matxData(rows * cols * sizeof(DTYPE));
		socket.recv(&matxData);

		char* matxBuffer = new char[matxData.size()];
		std::memcpy(matxBuffer, matxData.data(), matxData.size());

		Matrix m(rows, cols, matxBuffer);

		return m;
	}
}

void sendMatrix(Matrix& response, int32_t resType, zmq::socket_t& socket, bool duplicate, int32_t i) {
	if (!duplicate) {
		zmq::message_t header(HEADER_SIZE);
		populateHeader((char*)header.data(), OP::PUSH, i, response.rows, response.cols);
		socket.send(header, ZMQ_SNDMORE);
	}

	zmq::message_t matxData(response.getDataSize());
	std::memcpy(matxData.data(), response.getData(), response.getDataSize());

	if (!duplicate) {
		socket.send(matxData, ZMQ_SNDMORE);
	} else {
		socket.send(matxData);
	}
}

Matrix dot(Matrix& features, Matrix& weights) {
	int m = features.rows, k = features.cols, n = weights.cols;
	Matrix result(m, n);

	auto resultData = std::unique_ptr<DTYPE[]>(new DTYPE[m * n]);
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0,
		features.getData(), k, weights.getData(), n, 0.0, resultData.get(), n);

	result.setData(std::move(resultData));

	return result;
}

// Maybe not ideal but much more efficient since operating
// on contiguous data
Matrix activate(Matrix& mat) {
	DTYPE* activationData = new DTYPE[mat.rows * mat.cols];
	DTYPE* zData = mat.getData();
	
	for (int i = 0; i < mat.rows * mat.cols; ++i) {
		activationData[i] = std::tanh(zData[i]);
	}

	return Matrix(mat.rows, mat.cols, activationData);
}

invocation_response matmul(std::string dataserver, std::string weightserver, std::string dport, std::string wport, int32_t id, int32_t layer) {
	zmq::context_t ctx(1);
	char identity[sizeof(int32_t)];
	memcpy(&identity, (char*)&id, sizeof(int32_t));

	Timer getWeightsTimer;
	Timer getFeatsTimer;
	Timer computationTimer;
	Timer activationTimer;
	Timer sendResTimer;

	try {
		Matrix weights;
		std::thread t([&] {
			getWeightsTimer.start();
			zmq::socket_t weights_socket(ctx, ZMQ_DEALER);
			weights_socket.setsockopt(ZMQ_IDENTITY, identity, sizeof(int32_t));
			char whost_port[50];
			sprintf(whost_port, "tcp://%s:%s", weightserver.c_str(), wport.c_str());
			weights_socket.connect(whost_port);
			weights = requestMatrix(weights_socket, layer);
			getWeightsTimer.stop();
		});

		getFeatsTimer.start();
		zmq::socket_t data_socket(ctx, ZMQ_DEALER);
		data_socket.setsockopt(ZMQ_IDENTITY, identity, sizeof(int32_t));
		char dhost_port[50];
		sprintf(dhost_port, "tcp://%s:%s", dataserver.c_str(), dport.c_str());
		data_socket.connect(dhost_port);
		Matrix feats = requestMatrix(data_socket, id);
		getFeatsTimer.stop();
		t.join();

		if (weights.empty()) {
			return invocation_response::failure("Weights could not be loaded", "application/json");
		}
		if (feats.empty()) {
			return invocation_response::failure("No chunk corresponding to request", "appliation/json");
		}

		computationTimer.start();
		Matrix z = dot(feats, weights);
		computationTimer.stop();

		activationTimer.start();
		Matrix activations = activate(z);
		activationTimer.stop();

		sendResTimer.start();
		sendMatrix(z, 0, data_socket, false, id);
		sendMatrix(activations, 1, data_socket, true, id);
		sendResTimer.stop();
	} catch(std::exception &ex) {
		return invocation_response::failure(ex.what(), "application/json");
	}

	// Couldn't parse JSON with AWS SDK from ptree
	// For now creating a string with the times to be parsed on server
	std::string res = std::to_string(id) + ": " + std::to_string(getWeightsTimer.getTime()) + " " + std::to_string(getFeatsTimer.getTime())  + " " + std::to_string(computationTimer.getTime()) + " " + std::to_string(activationTimer.getTime()) + " " + std::to_string(sendResTimer.getTime());

	return invocation_response::success(res, "application/json");
}

static invocation_response my_handler(invocation_request const& request) {
	ptree pt;
	std::istringstream is(request.payload);
	read_json(is, pt);

	std::string dataserver = pt.get<std::string>("dataserver");
	std::string weightserver = pt.get<std::string>("weightserver");
	std::string dport = pt.get<std::string>("dport");
	std::string wport = pt.get<std::string>("wport");
	int32_t layer = pt.get<int32_t>("layer");
	int32_t chunkId = pt.get<int32_t>("id");

	return matmul(dataserver, weightserver, dport, wport, chunkId, layer);
}

int main(int argc, char *argv[]) {
	run_handler(my_handler);

	return 0;
}
