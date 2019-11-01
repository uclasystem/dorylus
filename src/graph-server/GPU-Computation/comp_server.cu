#include "comp_server.cuh"
static void doNotFreeBuffer(void *data, void *hint){
    // meant to be empty
    // printf("Buffer is not freed :)\n");
}
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

ComputingServer::ComputingServer(GPUComm* gpu_comm){
    nodeId=gpu_comm->nodeId;
    msgService=MessageService(gpu_comm->ctx, gpu_comm->dPort, gpu_comm->wPort);
    loadWeightServers(weightServerAddrs,gpu_comm->wServersFile);
    msgService.setUpWeightSocket(weightServerAddrs.at(nodeId%weightServerAddrs.size()));
}

//Start listening to main thread
void ComputingServer::run(){   
    // Keeps listening on coord's requests.
    try {
        bool terminate=0;
        int op;
        while (!terminate) {
            printLog(nodeId,"Receiving Next OP\n");
            op=msgService.requestFourBytes<int>();
            switch (op){
                case OP::REQ_FORWARD:
                    processForward();
                    break;
                case OP::REQ_BACKWARD:
                    processBackward();
                    break;
                case OP::TERM:
                    terminate=1;
                    msgService.terminateWeightServers(weightServerAddrs);
                    break;
                default:
                    printLog(nodeId,"Unknown OP\n");
            }
        }
    } catch (std::exception& ex) {
        std::cerr << "[ERROR] " << ex.what() << std::endl;
    }
}


void ComputingServer::processForward(){
    unsigned layer = msgService.requestFourBytes<unsigned>();
    unsigned lastLayer= msgService.requestFourBytes<unsigned>();
    // float split= msgService.requestFourBytes<float>();
    Matrix feats=msgService.requestMatrix();
    Matrix weights=msgService.requestWeightsMatrix(layer);
    FeatType* z_data=msgService.requestResultPtr();
    FeatType* act_z=msgService.requestResultPtr();
    CuMatrix z = cu.dot(feats, weights);
    memcpy(z_data,z.getData(),z.getDataSize());
    if(!lastLayer){
        cu.activate(z);//z data get activated ...
        memcpy(act_z,z.getData(),z.getDataSize());
    }else{
        CuMatrix cuPredictions=cu.softmaxRows(z);
        cuPredictions.updateMatrixFromGPU();
        memcpy(act_z,cuPredictions.getData(),z.getDataSize());
        delete[] cuPredictions.getData();
    }
    unsigned done= msgService.requestFourBytes<unsigned>();
    delete[] z.getData();
    delete[] weights.getData();
    // if(split!=0)
    //     evaluateModel(z);
}


void ComputingServer::evaluateModel(Matrix& activations){
    CuMatrix labels = cu.wrapMatrix(msgService.requestMatrix());
    CuMatrix cuAct =cu.wrapMatrix(activations);
    CuMatrix cuPredictions = cu.softmaxRows(cuAct);

    // Check if the label with the highest probability after softmax is equal to the
    // target label
    unsigned totalCorrect = cu.checkAccuracy(cuPredictions, labels);

    // Sum the individual losses of each vertex for this validation partition
    float lossThisPart = cu.checkLoss(cuPredictions, labels);

    msgService.sendFourBytes((char*)&totalCorrect);
    msgService.sendFourBytes((char*)&lossThisPart);
}

void ComputingServer::processBackward(){
    
    unsigned layer = msgService.requestFourBytes<unsigned>();
    unsigned numNode= msgService.requestFourBytes<unsigned>();
    unsigned lastLayer=msgService.requestFourBytes<unsigned>();

    //send INFO to weight server
    if(nodeId<weightServerAddrs.size()){
        unsigned count = 0; 
        for (size_t i=0;i<numNode;++i){
            if(i%weightServerAddrs.size()==nodeId)
                count+=1;
        }
        msgService.sendInfoMessage(count);
    }

    Matrix resultGradient;
    if (lastLayer) {
        std::cout << "< BACKWARD > Computing gradient from loss" << std::endl;
        gradLoss(layer);
    } else {
        std::cout << "< BACKWARD > Computing gradient for this layer" << std::endl;
        gradLayer(layer);
    }
    unsigned done= msgService.requestFourBytes<unsigned>();
}

void
ComputingServer::gradLayer(unsigned layer) {

    std::cout << "< BACKWARD > Requesting gradient from graph server" << std::endl;
    Matrix grad = msgService.requestMatrix();
    CuMatrix cuGrad=cu.wrapMatrix(grad);
    std::cout << "< BACKWARD > Requesting Z values" << std::endl;
    Matrix z = msgService.requestMatrix();
    CuMatrix cuZ = cu.wrapMatrix(z);

    std::cout << "< BACKWARD > Calculating derivative of activation" << std::endl;
    CuMatrix actDeriv = cu.activateDerivative(cuZ);
    std::cout << "< BACKWARD > Hadamard multiplication" << std::endl;
    CuMatrix interGrad = cu.hadamardMul(cuGrad,actDeriv);

    std::cout << "< BACKWARD > Getting weights" << std::endl;
    Matrix weights = msgService.requestWeightsMatrix(layer, OP::PULL_BACKWARD);
    std::cout << "< BACKWARD > MatMul(gradient, weights)" << std::endl;
    CuMatrix cuWeights=cu.wrapMatrix(weights);
    CuMatrix resultGrad = interGrad.dot(cuWeights, false, true);
    resultGrad.setData(msgService.requestResultPtr());
    resultGrad.updateMatrixFromGPU();

    std::cout << "< BACKWARD > Requesting AH" << std::endl;
    Matrix ah = msgService.requestMatrix();
    CuMatrix cuAh=cu.wrapMatrix(ah);
    std::cout << "< BACKWARD > Computing weight updates" << std::endl;
    CuMatrix cuWeightUpdates = cuAh.dot(interGrad, true, false, LEARNING_RATE);
    Matrix weightUpdates=cuWeightUpdates.getMatrix();

    std::cout << "< BACKWARD > Sending weight updates" << std::endl;
    msgService.sendWeightUpdate(weightUpdates,layer);
}


void
ComputingServer::gradLoss(unsigned layer) {
    std::cout << "< BACKWARD > Getting predictions and labels" << std::endl;
    Matrix predictions = msgService.requestMatrix();
    Matrix labels = msgService.requestMatrix();

    // derivative of softmax
    std::cout << "< BACKWARD > Calculating cross entropy" << std::endl;
    CuMatrix cuPredictions=cu.wrapMatrix(predictions);

    CuMatrix cuLabels=cu.wrapMatrix(labels);
    CuMatrix d_output = cu.hadamardSub(cuPredictions, cuLabels);

    // d_out * W^T
    std::cout << "< BACKWARD > Getting weights" << std::endl;
    Matrix weights = msgService.requestWeightsMatrix(layer, OP::PULL_BACKWARD);
    CuMatrix cuWeights=cu.wrapMatrix(weights);
    std::cout << "< BACKWARD > Computing gradient" << std::endl;
    CuMatrix interGrad = d_output.dot(cuWeights, false, true);
    interGrad.setData(msgService.requestResultPtr());
    interGrad.updateMatrixFromGPU();

    // AH^T * d_out
    std::cout << "< BACKWARD > Requesting AH" << std::endl;
    Matrix ah = msgService.requestMatrix();
    std::cout << "< BACKWARD > Computing weight updates" << std::endl;
    CuMatrix cuAh=cu.wrapMatrix(ah);
    CuMatrix cuWeightUpdates = cuAh.dot(d_output, true, false, LEARNING_RATE);
    Matrix weightUpdates=cuWeightUpdates.getMatrix();
    // std::cout<<weightUpdates.str();

    std::cout << "< BACKWARD > Sending weight updates" << std::endl;
    msgService.sendWeightUpdate(weightUpdates,layer);
}

/**
 *
 * Request the graph feature matrices data from dataserver.
 * 
 */ 

GraphData
MessageService::requestForwardMatrices(unsigned numLayers) {
        
    // Send pull request.
    // zmq::message_t header(HEADER_SIZE);
    // populateHeader((char *) header.data(), OP::PULL_BACKWARD, 0);
    GraphData graphData;

    // Receive z matrices chunks, from layer 1 -> last.
    for (size_t i = 1; i <= numLayers; ++i) 
        graphData.zMatrices.push_back(requestMatrix());

    // Receive act matrices chunks, from layer 0 -> last.
    for (size_t i = 0; i <= numLayers; ++i) 
        graphData.actMatrices.push_back(requestMatrix());

    // // Receive target label matrix chunk.
    graphData.targetMatrix = requestMatrix() ;

    return graphData;
}


void
MessageService::sendWeightUpdate(Matrix& matrix,unsigned layer) {
    // Send push header.
    zmq::message_t header(HEADER_SIZE);
    populateHeader((char *) header.data(), OP::PUSH_BACKWARD, layer, matrix.getRows(),
                    matrix.getCols());
    weightSocket->send(header, ZMQ_SNDMORE);

    zmq::message_t updateMsg(matrix.getData(),matrix.getDataSize(),doNotFreeBuffer,NULL);
    weightSocket->send(updateMsg);

    // Wait for updates settled reply.
    zmq::message_t confirm;
    weightSocket->recv(&confirm);
}

MessageService::MessageService(zmq::context_t& dctx,unsigned dPort_,unsigned wPort_):
    wctx(1),
    wPort(wPort_),
    wsocktReady(0),
    confirm(5){

    weightSocket=new zmq::socket_t(wctx, ZMQ_DEALER);
    dataSocket=new zmq::socket_t(dctx, ZMQ_REP);
    char ipc_addr[50];
    //use port as inproc communication addresss
    sprintf(ipc_addr, "inproc://%u", dPort_); 
    printLog(nodeId,"[GPU] Binding computing server to %s...\n" ,ipc_addr);
    dataSocket->bind(ipc_addr);
}

void MessageService::setUpWeightSocket(char* addr){
    wsocktReady=1;
    char ipc_addr[50];
    unsigned ipc_addr_len=strlen(ipc_addr);
    size_t identity_len = sizeof(unsigned) + ipc_addr_len;
    char identity[identity_len];
    memcpy(identity, (char *) &nodeId, sizeof(unsigned));
    memcpy(identity + sizeof(unsigned), ipc_addr, ipc_addr_len);
    weightSocket->setsockopt(ZMQ_IDENTITY, identity, identity_len);
    char whost_port[50];
    sprintf(whost_port, "tcp://%s:%u", addr, wPort);
    printf("connect to %s\n", whost_port);
    weightSocket->connect(whost_port);
}

template <class T>
T MessageService::requestFourBytes(){
    zmq::message_t header(4);
    dataSocket->recv(&header);
    dataSocket->send(confirm);
    return *((T*)header.data());
}


void MessageService::sendFourBytes(char* data){
    zmq::message_t dataMsg(4);
    memcpy(dataMsg.data(),data,4);
    dataSocket->send(dataMsg);
    dataSocket->recv(&confirm);
}

Matrix MessageService::requestMatrix(){
    zmq::message_t matrixInfo;
    dataSocket->recv(&matrixInfo);
    dataSocket->send(confirm);
    unsigned row=parse<unsigned>((char *) matrixInfo.data(), 0);
    unsigned col=parse<unsigned>((char *) matrixInfo.data(), 1);
    FeatType* data;
    std::memcpy(&data,(char*)matrixInfo.data()+2*sizeof(unsigned),sizeof(FeatType*));
    return Matrix(row,col,data);
}


FeatType* MessageService::requestResultPtr(){
    zmq::message_t ptrMsg(sizeof(FeatType*));
    dataSocket->recv(&ptrMsg);
    dataSocket->send(confirm);
    return *((FeatType**)ptrMsg.data());
}   

Matrix MessageService::requestWeightsMatrix(unsigned layer,OP op){
    // Send pull request.
    zmq::message_t header(HEADER_SIZE);
    populateHeader((char *) header.data(), op, layer);
    weightSocket->send(header);
    // Listen on respond.
    zmq::message_t respHeader(HEADER_SIZE);
    weightSocket->recv(&respHeader);
    // Parse the respond.
    unsigned layerResp = parse<unsigned>((char *) respHeader.data(), 1);
    if ((int)layerResp == -1) {      // Failed.
        std::cerr << "[ ERROR ] No corresponding matrix chunk!" << std::endl;
        return Matrix();
    } else {                    // Get matrices data.
        unsigned rows = parse<unsigned>((char *) respHeader.data(), 2);
        unsigned cols = parse<unsigned>((char *) respHeader.data(), 3);
        zmq::message_t wData(rows * cols * sizeof(FeatType));
        weightSocket->recv(&wData);
        FeatType *wBuffer = new FeatType[rows*cols];
        memcpy((char*)wBuffer,(char*)wData.data(),rows * cols * sizeof(FeatType));
        Matrix m(rows, cols, wBuffer);
        return m;
    }
}

void MessageService::sendInfoMessage(unsigned numLambdas) {
    zmq::message_t header(HEADER_SIZE);
    populateHeader((char *) header.data(), OP::INFO, numLambdas);
    weightSocket->send(header);

    // Wait for info received reply.
    zmq::message_t confirm;
    weightSocket->recv(&confirm);
}

std::vector<Matrix> 
MessageService::requestWeightsMatrices(unsigned numLayers){
    // Send pull request.
    zmq::message_t header(HEADER_SIZE);
    populateHeader((char *) header.data(), OP::PULL_BACKWARD, 0);
    weightSocket->send(header);

    std::vector<Matrix> weightsData;

    // Receive weight matrices, from layer 2 -> last.
    for (size_t i = 2; i <= numLayers; ++i) {
        zmq::message_t respHeader;
        weightSocket->recv(&respHeader);
        unsigned layerResp = parse<unsigned>((char *) respHeader.data(), 1);
        if (layerResp == ERR_HEADER_FIELD) {    // Failed.
            std::cerr << "[ ERROR ] No corresponding weight matrix!" << std::endl;
            return weightsData;
        } else {                    // Get matrices data.
            unsigned rows = parse<unsigned>((char *) respHeader.data(), 2);
            unsigned cols = parse<unsigned>((char *) respHeader.data(), 3);
            zmq::message_t matxData(rows * cols * sizeof(FeatType));
            weightSocket->recv(&matxData);

            FeatType *matxBuffer = new FeatType[rows * cols];
            std::memcpy(matxBuffer, matxData.data(), matxData.size());

            weightsData.push_back(Matrix(rows, cols, matxBuffer));
        }
    }
    return weightsData;
}

void 
MessageService::terminateWeightServers(std::vector<char*>& weightServerAddrs){
    if(nodeId!=0)
        return; 
    
    printLog(nodeId,"Node 0 is terminating all weightservers\n");

    for (unsigned i = 0; i < weightServerAddrs.size(); ++i) {
        zmq::socket_t ws=zmq::socket_t(wctx, ZMQ_DEALER);
        char identity[] = "coordx";
        ws.setsockopt(ZMQ_IDENTITY, identity, strlen(identity) + 1);
        char whost_port[50];
        sprintf(whost_port, "tcp://%s:%u", weightServerAddrs[i], wPort);
        printLog(nodeId,"[GPU]Shutting Down Weightserver %s \n", whost_port);
        ws.connect(whost_port);
        sendShutdownMessage(ws);
        ws.close();
        // free(weightserverAddrs[i]); 
    }
}

void
MessageService::sendShutdownMessage(zmq::socket_t& weightsocket) {
    zmq::message_t header(HEADER_SIZE);
    populateHeader((char *) header.data(), OP::TERM);
    weightSocket->send(header);
    
    // Set receive timeou 1s property on this weightsocket, in case that a weightserver is dying too quickly that it's
    // confirm message it not sent from buffer yet. Using timeout here because shutdown is not a big deal.
    weightSocket->setsockopt(ZMQ_RCVTIMEO, 1000);

    // Wait for termination confirmed reply.
    zmq::message_t confirm;
    weightSocket->recv(&confirm);
}


