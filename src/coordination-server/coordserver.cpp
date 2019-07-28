#include <aws/core/Aws.h>
#include <aws/core/utils/json/JsonSerializer.h>
#include <aws/core/utils/Outcome.h>
#include <aws/core/utils/logging/DefaultLogSystem.h>
#include <aws/core/utils/logging/AWSLogging.h>
#include <aws/core/utils/HashingUtils.h>
#include <aws/lambda/LambdaClient.h>
#include <aws/lambda/model/InvokeRequest.h>

#include <chrono>
#include <iostream>
#include <fstream>
#include <mutex>
#include <sstream>
#include <thread>
#include <vector>
#include <sys/stat.h>
#include <unistd.h>
#include <zmq.hpp>


std::mutex m;
std::condition_variable cv;
int cnt = 0;
int total = 0;


static const char *ALLOCATION_TAG = "matmulLambda";
static std::shared_ptr<Aws::Lambda::LambdaClient> m_client;


using namespace std::chrono;


/** Parse the serialized ZMQ message header. */
template<class T>
T parse(const char* buf, int32_t offset) {
	T val;
	std::memcpy(&val, buf + (offset * sizeof(T)), sizeof(T));
	return val;
}


/**
 *
 * Callback function to be called after receiving the respond from lambda threads.
 * 
 */
void
callback(const Aws::Lambda::LambdaClient *client, const Aws::Lambda::Model::InvokeRequest &invReq, const Aws::Lambda::Model::InvokeOutcome &outcome,
	     const std::shared_ptr<const Aws::Client::AsyncCallerContext> &context) {
	
	// Lambda returns success
	if (outcome.IsSuccess()) {
		Aws::Lambda::Model::InvokeResult& result = const_cast<Aws::Lambda::Model::InvokeResult&>(outcome.GetResult());

		// JSON Parsing not working from Boost to AWS.
		Aws::IOStream& payload = result.GetPayload();
		Aws::String functionResult;
		std::getline(payload, functionResult);

		// No error found means a successful respond.
		if (functionResult.find("error") == std::string::npos) {
			std::string num = functionResult.substr(0, functionResult.find(":")).c_str();
			std::cout << "\033[1;32m[SUCCESS]\033[0m\t" << functionResult << std::endl;
			
		// There is error in the results.
		} else
			std::cout << "\033[1;31m [ERROR] \033[0m\t" << functionResult << std::endl;

	// Lambda returns error.
	} else {
		Aws::Lambda::Model::InvokeResult& result = const_cast<Aws::Lambda::Model::InvokeResult&>(outcome.GetResult());

		Aws::IOStream& payload = result.GetPayload();
		Aws::String functionResult;
		std::getline(payload, functionResult);
		std::cout << "\033[1;31m [ERROR] \033[0m\t" << functionResult << std::endl;
	}
}


/**
 *
 * Invoke a lambda function of the given named (previously registered on lambda cloud).
 * 
 */
void
invokeFunction(Aws::String funcName, char *dataserver, char* dport, char *weightserver, char *wport, int32_t layer, int32_t id) {
	Aws::Lambda::Model::InvokeRequest invReq;
	invReq.SetFunctionName(funcName);
	invReq.SetInvocationType(Aws::Lambda::Model::InvocationType::RequestResponse);
	invReq.SetLogType(Aws::Lambda::Model::LogType::Tail);
	std::shared_ptr<Aws::IOStream> payload = Aws::MakeShared<Aws::StringStream>("FunctionTest");
	Aws::Utils::Json::JsonValue jsonPayload;
	jsonPayload.WithString("dataserver", dataserver);
	jsonPayload.WithString("weightserver", weightserver);
	jsonPayload.WithString("wport", wport);
	jsonPayload.WithString("dport", dport);
	jsonPayload.WithInteger("layer", layer);
	jsonPayload.WithInteger("id", id);
	*payload << jsonPayload.View().WriteReadable();
	invReq.SetBody(payload);
	m_client->InvokeAsync(invReq, callback);
}


/**
 *
 * Coordination server keeps listening on the dataserver's request of issuing lambda threads.
 * 
 */
int
main(int argc, char *argv[]) {
	Aws::SDKOptions options;
	Aws::InitAPI(options);

	assert(argc == 5);

	// Setup ZeroMQ.
	zmq::context_t ctx(1);
	zmq::socket_t frontend(ctx, ZMQ_REP);
	char host_port[50];
	sprintf(host_port, "tcp://*:%s", argv[1]);
	std::cout << "Binding to dataserver port " << host_port << "..." << std::endl;
	frontend.bind(host_port);

	// Setup lambda client.
	Aws::Client::ClientConfiguration clientConfig;
	clientConfig.requestTimeoutMs = 900000;
	clientConfig.region = "us-east-2";
	m_client = Aws::MakeShared<Aws::Lambda::LambdaClient>(ALLOCATION_TAG, clientConfig);

	// Keeps listening on dataserver's requests.
	std::cout << "[Coord] Starts listening on dataserver's requests..." << std::endl;
	while (1) {
		zmq::message_t header;
		zmq::message_t dataserverIp;

		try {
			frontend.recv(&header);
			frontend.recv(&dataserverIp);

			zmq::message_t reply(3);
			std::memcpy(reply.data(), "ACK", 3);
			frontend.send(reply);
		} catch (std::exception& ex) {
			std::cerr << ex.what() << std::endl;
			return 13;
		}

		int32_t layer = parse<int32_t>((char *)header.data(), 1);
		int32_t nThreadsReq = parse<int32_t>((char *)header.data(), 2);

		std::string accMsg = "[ACCEPTED] Req for " + std::to_string(nThreadsReq) + " lambdas for layer " + std::to_string(layer);
		std::cout << accMsg << std::endl;
		total = nThreadsReq;

		// Get a request. Issue a bunch of lambda threads to serve the request.
		{
			for (int i = 0; i < nThreadsReq; i++)
				invokeFunction("matmul-cpp", (char *)dataserverIp.data(), argv[4], argv[2], argv[3], layer, i);
		}
	}

	m_client = nullptr;
	Aws::ShutdownAPI(options);

	return 0;
}
