#include <aws/core/Aws.h>
#include <aws/core/utils/json/JsonSerializer.h>
#include <aws/core/utils/Outcome.h>
#include <aws/core/utils/logging/DefaultLogSystem.h>
#include <aws/core/utils/logging/AWSLogging.h>
#include <aws/core/utils/HashingUtils.h>
#include <aws/lambda/LambdaClient.h>
#include <aws/lambda/model/InvokeRequest.h>
#include <ctype.h>
#include "coordserver.hpp"


using namespace std::chrono;


static const char *ALLOCATION_TAG = "GNN-Lambda";
static std::shared_ptr<Aws::Lambda::LambdaClient> m_client;
static std::ofstream outfile;


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
            std::cout << "\033[1;32m[SUCCESS]\033[0m\t" << functionResult << std::endl;
            outfile << functionResult << std::endl;

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
               unsigned layer, unsigned id, bool backward, bool lastLayer) {
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
    jsonPayload.WithInteger("layer", layer);    // For forward-prop: layer-ID; For backward-prop: numLayers.
    jsonPayload.WithInteger("id", id);
    if (backward) {
        jsonPayload.WithBool("lastLayer", lastLayer);
    }
    *payload << jsonPayload.View().WriteReadable();
    invReq.SetBody(payload);
    m_client->InvokeAsync(invReq, callback);
}


/**
 *
 * CoordServer constructor.
 *
 */
CoordServer::CoordServer(char *coordserverPort_, char *weightserverFile_, char *weightserverPort_, char *dataserverPort_)
    : coordserverPort(coordserverPort_), weightserverFile(weightserverFile_),
      weightserverPort(weightserverPort_), dataserverPort(dataserverPort_), ctx(1) {
    loadWeightServers(weightserverAddrs,weightserverFile);
    std::cout << "Detected " << weightserverAddrs.size() << " weight servers to use." << std::endl;
}

/**
 *
 * Runs the coordserver, keeps listening on dataserver's requests for lambda threads invocation.
 *
 */
void
CoordServer::run() {

    // Bind dataserver socket.
    zmq::socket_t datasocket(ctx, ZMQ_REP);
    char dhost_port[50];
    sprintf(dhost_port, "tcp://*:%s", coordserverPort);
    std::cout << "Binding to dataserver " << dhost_port << "..." << std::endl;
    datasocket.bind(dhost_port);

    // Connect weightserver sockets. Since sockets on weightserver are DEALERs, so we also have to create DEALERs and
    // set a socket identity.
    std::vector<zmq::socket_t> weightsockets;
    std::cout << "Connecting to all weightservers..." << std::endl;
    for (unsigned i = 0; i < weightserverAddrs.size(); ++i) {
        weightsockets.push_back(zmq::socket_t(ctx, ZMQ_DEALER));
        char identity[] = "coord";
        weightsockets[i].setsockopt(ZMQ_IDENTITY, identity, strlen(identity) + 1);
        char whost_port[50];
        sprintf(whost_port, "tcp://%s:%s", weightserverAddrs[i], weightserverPort);
        std::cout << "  found weightserver " << whost_port << std::endl;
        weightsockets[i].connect(whost_port);
    }

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
            datasocket.recv(&header);
            datasocket.recv(&dataserverIp);

            // Send ACK confirm reply.
            zmq::message_t confirm;
            datasocket.send(confirm);

            // Parse the request.
            unsigned op = parse<unsigned>((char *) header.data(), 0);
            unsigned layer = parse<unsigned>((char *) header.data(), 1);
            unsigned nThreadsReq = parse<unsigned>((char *) header.data(), 2);

            // Append a terminating null char to ensure this is a valid C string.
            char* dataserverIpCopy = new char[dataserverIp.size() + 1];
            memcpy(dataserverIpCopy, (char *) dataserverIp.data(), dataserverIp.size());
            dataserverIpCopy[dataserverIp.size()] = '\0';

            // If it is a termination message, then shut all weightservers first and then shut myself down.
            if (op == OP::TERM) {
                std::cerr << "[SHUTDOWN] Terminating the servers..." << std::endl;
                terminate = true;
                for (unsigned i = 0; i < weightserverAddrs.size(); ++i) {
                    sendShutdownMessage(weightsockets[i]);
                    free(weightserverAddrs[i]);     // Free the `strdup`ed weightserver Ips.
                }

            // Else is a request for lambda threads. Handle that. This is forward.
            } else if (op == OP::REQ_FORWARD) {
                std::string accMsg = "[ACCEPTED] Req for FORWARD, lambda " + std::to_string(nThreadsReq)
                                     + " is invoked for layer " + std::to_string(layer) + ".";
                std::cout << accMsg << std::endl;

                unsigned lambdaId = nThreadsReq;
                // Issue the lambda thread to serve the request.
                char *weightserverIp = weightserverAddrs[req_count % weightserverAddrs.size()];
                invokeFunction("new-forward-prop", dataserverIpCopy, dataserverPort, weightserverIp, weightserverPort, layer, lambdaId, false, false);
                req_count++;

            // This is backward.
            } else if (op == OP::REQ_BACKWARD) {
                std::string accMsg = "[ACCEPTED] Req for BACKWARD " + std::to_string(nThreadsReq)
                                   + " lambdas for layer " + std::to_string(layer) + ".";
                std::cout << accMsg << std::endl;

                // Calculate numLambdas assigned to each weightserver.
                unsigned baseNumThreads = nThreadsReq / weightserverAddrs.size();
                std::vector<unsigned> numLambdas(weightserverAddrs.size(), baseNumThreads);
                for (unsigned i = 0; i < nThreadsReq % weightserverAddrs.size(); ++i)
                    numLambdas[(req_count + i) % weightserverAddrs.size()]++;

                // Send info message to weightservers.
                std::cout << "[  INFO  ] Sending number of lambdas info to weightservers..." << std::endl;
                for (unsigned i = 0; i < weightserverAddrs.size(); ++i)
                    sendInfoMessage(weightsockets[i], numLambdas[i]);

                // Issue a bunch of lambda threads to serve the request.
                for (unsigned i = 0; i < nThreadsReq; ++i) {
                    char *weightserverIp = weightserverAddrs[req_count % weightserverAddrs.size()];
                    invokeFunction("new-backprop", dataserverIpCopy, dataserverPort, weightserverIp, weightserverPort, layer, i, true, layer==2);
                    req_count++;
                }
                std::cout << "after invoke backward lambda\n";

            // Unknown op code.
            } else {
                std::cerr << "[ ERROR ] Unknown OP code (" << op << ") received." << std::endl;
            }

            delete[] dataserverIpCopy;
        }
    } catch (std::exception& ex) { /** Context Termintated. */ }
}


/**
 *
 * Load the weightservers configuration file.
 *
 */
void
CoordServer::loadWeightServers(std::vector<char *>& addresses, const std::string& wServersFile){
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


/**
 *
 * Send info message (of number of lambdas assigned to it) to a weightserver.
 *
 */
void
CoordServer::sendInfoMessage(zmq::socket_t& weightsocket, unsigned numLambdas) {
    zmq::message_t header(HEADER_SIZE);
    populateHeader((char *) header.data(), OP::INFO, numLambdas);
    weightsocket.send(header);

    // Wait for info received reply.
    zmq::message_t confirm;
    weightsocket.recv(&confirm);
}


/**
 *
 * Sends a shutdown message to all the weightservers.
 *
 */
void
CoordServer::sendShutdownMessage(zmq::socket_t& weightsocket) {
    zmq::message_t header(HEADER_SIZE);
    populateHeader((char *) header.data(), OP::TERM);
    weightsocket.send(header);

    // Set receive timeou 1s property on this weightsocket, in case that a weightserver is dying too quickly that it's
    // confirm message it not sent from buffer yet. Using timeout here because shutdown is not a big deal.
    weightsocket.setsockopt(ZMQ_RCVTIMEO, 500);

    // Wait for termination confirmed reply.
    zmq::message_t confirm;
    weightsocket.recv(&confirm);
}


/** Main entrance: Starts a coordserver instance and run a single listener, until termination msg received. */
int
main(int argc, char *argv[]) {

    assert(argc == 6);

    char *coordserverPort = argv[1];
    char *weightserverFile = argv[2];
    char *weightserverPort = argv[3];
    char *dataserverPort = argv[4];

    // Set output file location.
    std::string outfileName = std::string(argv[5]) + "/output";
    outfile.open(outfileName, std::fstream::out);
    assert(outfile.good());

    Aws::SDKOptions options;
    Aws::InitAPI(options);

    CoordServer cs(coordserverPort, weightserverFile, weightserverPort, dataserverPort);
    cs.run();

    m_client = nullptr;
    Aws::ShutdownAPI(options);

    return 0;
}
