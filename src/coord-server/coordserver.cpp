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
#include <memory>
#include <sstream>
#include <thread>
#include <vector>
#include <sys/stat.h>
#include <unistd.h>
#include <zmq.hpp>
#include "../utils/utils.hpp"
#include <boost/algorithm/string/trim.hpp>


static const char *ALLOCATION_TAG = "matmulLambda";
static std::shared_ptr<Aws::Lambda::LambdaClient> m_client;


using namespace std::chrono;


/**
 *
 * Callback function to be called after receiving the respond from lambda threads.
 * 
 */
static void
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
            std::cout << "\033[1;31m[ ERROR ]\033[0m\t" << functionResult << std::endl;

    // Lambda returns error.
    } else {
        Aws::Lambda::Model::InvokeResult& result = const_cast<Aws::Lambda::Model::InvokeResult&>(outcome.GetResult());

        Aws::IOStream& payload = result.GetPayload();
        Aws::String functionResult;
        std::getline(payload, functionResult);
        std::cout << "\033[1;31m[ ERROR ]\033[0m\t" << functionResult << std::endl;
    }
}


/**
 *
 * Invoke a lambda function of the given named (previously registered on lambda cloud).
 * 
 */
static void
invokeFunction(Aws::String funcName, char *dataserver, char *dport, char *weightserver, char *wport,
               unsigned layer, unsigned id) {
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
 * Class of the coordserver. Coordination server keeps listening on the dataserver's request of issuing lambda threads.
 * 
 */
class CoordServer {

public:

    CoordServer(char *coordserverPort_, char *weightserverFile_, char *weightserverPort_, char *dataserverPort_)
        : coordserverPort(coordserverPort_), weightserverFile(weightserverFile_),
          weightserverPort(weightserverPort_), dataserverPort(dataserverPort_) {
        loadWeightServers(weightserverAddrs,weightserverFile);
        std::cout << "Detected " << weightserverAddrs.size() << " weight servers to use." << std::endl;
    }

    // Runs the coordserver, keeps listening on dataserver's requests for lambda threads invocation.
    void run() {
        zmq::context_t ctx(1);
        zmq::socket_t frontend(ctx, ZMQ_REP);
        char host_port[50];
        sprintf(host_port, "tcp://*:%s", coordserverPort);
        std::cout << "Binding coordination server to " << host_port << "..." << std::endl;
        frontend.bind(host_port);

        // Setup lambda client.
        Aws::Client::ClientConfiguration clientConfig;
        clientConfig.requestTimeoutMs = 900000;
        clientConfig.region = "us-east-2";
        m_client = Aws::MakeShared<Aws::Lambda::LambdaClient>(ALLOCATION_TAG, clientConfig);

        // Keeps listening on dataserver's requests.
        std::cout << "[Coord] Starts listening for dataserver's requests..." << std::endl;
       
        try {
            bool terminate = false;
            size_t req_count = 0;
            while (!terminate) {

                // Wait on requests.
                zmq::message_t header;
                zmq::message_t dataserverIp;
                frontend.recv(&header);
                frontend.recv(&dataserverIp);

                // Send ACK confirm reply.
                zmq::message_t confirm;
                frontend.send(confirm);

                // Parse the request.
                unsigned op = parse<unsigned>((char *) header.data(), 0);
                unsigned layer = parse<unsigned>((char *) header.data(), 1);
                unsigned nThreadsReq = parse<unsigned>((char *) header.data(), 2);

                char dataserverIpCopy[dataserverIp.size() + 1];
                memcpy(dataserverIpCopy, (char *) dataserverIp.data(), dataserverIp.size());
                dataserverIpCopy[dataserverIp.size()] = '\0';

                // If it is a termination message, then shut all weightservers first and then shut myself down.
                if (op == OP::TERM) {
                    std::cerr << "Terminating the servers..." << std::endl;
                    terminate = true;
                    for (std::size_t i = 0; i < weightserverAddrs.size(); ++i)
                    	sendShutdownMessage(weightserverAddrs[i], weightserverPort);

                // Else is a pull request for weight matrix. Handle that.
                } else {
                    std::string accMsg = "[ACCEPTED] Req for " + std::to_string(nThreadsReq)
                                       + " lambdas for layer " + std::to_string(layer);
                    std::cout << accMsg << std::endl;

                    // Issue a bunch of lambda threads to serve the request.
                    for (unsigned i = 0; i < nThreadsReq; i++) {

                        // TODO: Maybe improve this naive round robin scheduling.
                    	char *weightserverIp = weightserverAddrs[req_count % weightserverAddrs.size()];
                        invokeFunction("forward-prop-cpp", dataserverIpCopy, dataserverPort, weightserverIp, weightserverPort, layer, i);
                   		req_count++;
					}
                }
            }
        } catch (std::exception& ex) {
            std::cerr << "[ERROR] " << ex.what() << std::endl;
        }
    }

private:

	// Load the weightservers configuration file.
	void loadWeightServers(std::vector<char *>& addresses, const std::string& wServersFile){
		std::ifstream infile(wServersFile);
		if (!infile.good())
	        printf("Cannot open weight server file: %s [Reason: %s]\n", wServersFile.c_str(), std::strerror(errno));

	    assert(infile.good());

	    std::string line;
	    while (!infile.eof()) {
	        std::getline(infile, line);
	        boost::algorithm::trim(line);

	        if (line.length() == 0)
	        	continue;	
            
	    	char *addr = strdup(line.c_str());
	    	addresses.push_back(addr);
	    }
	}


    // Sends a shutdown message to all the weightservers.

    void sendShutdownMessage(char *weightserverPort, char *weightserverIp) {
        char identity[] = "coord";
        char wHostPort[50];
        sprintf(wHostPort, "tcp://%s:%s", weightserverPort, weightserverIp);

        zmq::context_t ctx(1);
        zmq::socket_t weightsocket(ctx, ZMQ_DEALER);
        weightsocket.setsockopt(ZMQ_IDENTITY, identity, sizeof(identity));

        zmq::message_t header(HEADER_SIZE);
        populateHeader((char *) header.data(), OP::TERM);
        weightsocket.connect(wHostPort);
        weightsocket.send(header);
    }

    char *coordserverPort;
    char *weightserverFile;
    char *weightserverPort;
    char *dataserverPort;
    std::vector<char*> weightserverAddrs;
};


/** Main entrance: Starts a coordserver instance and run. */
int
main(int argc, char *argv[]) {
    assert(argc == 5);
    printf("hello\n");
    char *coordserverPort = argv[1];
    char *weightserverFile = argv[2];
    char *weightserverPort = argv[3];
    char *dataserverPort = argv[4];

    Aws::SDKOptions options;
    Aws::InitAPI(options);

    CoordServer cs(coordserverPort, weightserverFile, weightserverPort, dataserverPort);
    cs.run();

    m_client = nullptr;
    Aws::ShutdownAPI(options);

    return 0;
}
