#include "comp_server.cuh"

unsigned getMaxIndex(FeatType* row, unsigned length) {
    float max = 0.0;
    unsigned maxIndex = 0;
    for (unsigned col = 0; col < length; ++col) {
        if (row[col] > max) {
            max = row[col];
            maxIndex = col;
        }
    }

    return maxIndex;
}
unsigned getLabelIndex(FeatType* row, unsigned length) {
    for (unsigned col = 0; col < length; ++col) {
        if (row[col] == 1)
            return col;
    }

    // Should never get here
    return (unsigned)-1;
}

static void doNotFreeBuffer(void *data, void *hint){
    // printf("Buffer is not freed :)\n");
}

ComputingServer::ComputingServer(zmq::context_t& dctx,unsigned dPort_,const std::string& wServersFile,unsigned wPort_):
    dPort(dPort_),
    wPort(wPort_),
    dataSocket(dctx, ZMQ_REP),
    weightSocket(wctx, ZMQ_DEALER){
        loadWeightServers(weightServerAddrs,wServersFile);

        //use port as ipc addresss
        sprintf(ipc_addr, "inproc://%u", dPort); 
        printLog(nodeId,"Binding computing server to %s...\n" ,ipc_addr);
        dataSocket.bind(ipc_addr);
        // server_ready=1;
        // cv.notify_one();
}

void ComputingServer::evaluateModel(Matrix& activations){
    CuMatrix labels = cu.wrapMatrix(requestTargetMatrix());
    CuMatrix cuPredictions = cu.softmaxRows(labels);

    // Check if the label with the highest probability after softmax is equal to the
    // target label
    unsigned totalCorrect = cu.checkAccuracy(cuPredictions, labels);

    // Sum the individual losses of each vertex for this validation partition
    float lossThisPart = cu.checkLoss(cuPredictions, labels);

    zmq::message_t header(HEADER_SIZE);
    populateHeader((char*)header.data(), OP::PUSH_EVAL, 0, totalCorrect);
    serialize<float>((char*)header.data(), 3, lossThisPart);
    dataSocket.send(header);

}

Matrix ComputingServer::requestTargetMatrix(){
    // zmq::message_t header(HEADER_SIZE);
    // populateHeader((char *) header.data(), OP::PULL_EVAL, 0);
    // dataSocket.send(header);
    zmq::message_t respHeader;
    dataSocket.recv(&respHeader);


    unsigned layerResp = parse<unsigned>((char *) respHeader.data(), 1);
    unsigned rows = parse<unsigned>((char *) respHeader.data(), 2);
    unsigned cols = parse<unsigned>((char *) respHeader.data(), 3);
    zmq::message_t matxData;
    dataSocket.recv(&matxData);
    FeatType *matxBuffer = new FeatType[rows * cols];
    std::memcpy(matxBuffer, matxData.data(), matxData.size());
;

    Matrix m(rows, cols, matxBuffer);
    return m;
}


unsigned ComputingServer::checkAccuracy(Matrix& predictions, Matrix& labels){
    unsigned totalCorrect = 0;
    unsigned length = predictions.getCols();
    for (unsigned r = 0; r < predictions.getRows(); ++r) {
        unsigned maxIndex = getMaxIndex(predictions.get(r), length);
        if (labels.get(r, maxIndex) == 1.0)
            ++totalCorrect;
    }

    return totalCorrect;

}

float ComputingServer::checkLoss(Matrix& preds, Matrix& labels){
    assert(preds.getRows() == labels.getRows());
    assert(preds.getCols() == labels.getCols());

    float totalLoss = 0;
    unsigned length = preds.getCols();
    for (unsigned r = 0; r < preds.getRows(); ++r) {
        unsigned labelIndex = getLabelIndex(labels.get(r), length);
        float lossThisRow = -(std::log(preds.get(r, labelIndex)));
        totalLoss += lossThisRow;
    }

    return totalLoss;

}

void ComputingServer::run(){
    
    // Keeps listening on coord's requests.
    printLog(1,"[GPU] Starts listening for GPU requests from DATASERVER...\n");

    zmq::message_t confirm(5);
    zmq::message_t init_header(HEADER_SIZE);
    dataSocket.recv(&init_header);
    printLog(1,"[GPU] Receved...\n");    
    nodeId = parse<unsigned>((char *) init_header.data(), 0);
    dataSocket.send(confirm);

    try {
        bool terminate=0;
        while (!terminate) {
            printLog(nodeId,"Receiving Next OP\n");
            zmq::message_t header(HEADER_SIZE);
            dataSocket.recv(&header);
            unsigned op = parse<unsigned>((char *) header.data(), 0);
            dataSocket.send(confirm);

            switch (op){
                case OP::TERM:
                    terminate=1;
                    terminateWeightServers();
                    break;
                case OP::REQ_FORWARD:
                    processForward(header);
                    break;
                case OP::REQ_BACKWARD:
                    processBackward(header);
                    break;
                default:
                    printLog(nodeId,"Unknown OP\n");
            }
        }
    } catch (std::exception& ex) {
        std::cerr << "[ERROR] " << ex.what() << std::endl;
    }
}


void ComputingServer::sendInfoMessage(zmq::socket_t& weightsocket, unsigned numLambdas) {
    zmq::message_t header(HEADER_SIZE);
    populateHeader((char *) header.data(), OP::INFO, numLambdas);
    weightsocket.send(header);

    // Wait for info received reply.
    zmq::message_t confirm;
    weightsocket.recv(&confirm);
}

void ComputingServer::processBackward(zmq::message_t &header){
    
    zmq::message_t confirm(5);
    unsigned layer= parse<unsigned>((char *) header.data(), 1);
    unsigned numNode = parse<unsigned>((char *) header.data(), 2);

    double t1=getTimer();
    //send INFO to weight server
    if(nodeId<weightServerAddrs.size()){
        unsigned count = 0; 
        for (size_t i=0;i<numNode;++i)
            if(i%weightServerAddrs.size()==nodeId)
                count+=1;
        sendInfoMessage(weightSocket, count);
    }

    std::vector<Matrix> weightsData;
    // Request weights matrices.
    std::thread t([&] {     // Weight requests run in a separate thread.
        double t2=getTimer();
        std::cout << "< BACKWARD > Asking weightserver..." << std::endl;
        weightsData =  requestWeightsMatrices(layer);
        std::cout << "< BACKWARD > Got data from weightserver." << std::endl;
        printf("BACKWARD Weight Fetch Time*: %lf\n", getTimer()-t2);

    });
    double t3=getTimer();
    GraphData graphData=requestForwardMatrices(layer);
    printf("BACKWARD Features Fetch Time*: %lf\n", getTimer()-t3);
    t.join();
    std::cout << "< BACKWARD > Doing the gradient descent computation..." << std::endl;
    
    std::vector<Matrix> weightsUpdates;
    weightsUpdates = gradientComputation(graphData, weightsData);
    printf("BACKWARD Computation Time*: %lf\n", getTimer()-t1);

    sendWeightsUpdates(weightsUpdates);
}

void
ComputingServer::sendWeightsUpdates(std::vector<Matrix> weightsUpdates) {
    
    // Send push header.
    zmq::message_t header(HEADER_SIZE);
    populateHeader((char *) header.data(), OP::PUSH_BACKWARD, 0);
    printf("Sending updates\n");
    weightSocket.send(header, ZMQ_SNDMORE);

    // Send updates to all weight matrices given by my chunk.
    for (unsigned i = 0; i < weightsUpdates.size(); ++i) {
        Matrix& updateMat = weightsUpdates[i];

        zmq::message_t updateData(updateMat.getData(),updateMat.getDataSize(),doNotFreeBuffer,NULL);
        if (i == weightsUpdates.size() - 1)
            weightSocket.send(updateData);
        else
            weightSocket.send(updateData, ZMQ_SNDMORE);
    }

    // Wait for updates settled reply.
    zmq::message_t confirm;
    weightSocket.recv(&confirm);
}


/**
 *
 * Main logic of gradient computation and a naive gradient descent to get weight updates.
 *
 * Attention:
 *   zMatrices   vec contains z1   -> zout;
 *   actMatrices vec contains act0 -> actout;
 *   weightData  vec contains w2   -> wout.
 * 
 */
std::vector<Matrix>
ComputingServer::gradientComputation(GraphData& graphData, std::vector<Matrix>& weightsData) {
    
    std::vector<CuMatrix*> gradients;
    std::vector<Matrix> weightsUpdates;

    // Compute last layer's gradients.
    CuMatrix cuAct=cu.wrapMatrix(graphData.actMatrices.back());
    CuMatrix softmaxRes = cu.softmaxRows(cuAct);

    CuMatrix cuTarget=cu.wrapMatrix(graphData.targetMatrix);
    CuMatrix subRes = cu.hadamardSub(softmaxRes, cuTarget);

    CuMatrix cuZ=cu.wrapMatrix(graphData.zMatrices.back());
    CuMatrix derivateRes = cu.activateDerivate(cuZ);
    gradients.push_back(cu.hadamardMul(subRes, derivateRes));

    // Compute previous layers gradients.
    for (unsigned i = weightsData.size(); i > 0; --i) {
        CuMatrix cuWeights=cu.wrapMatrix(weightsData[i - 1]);
        CuMatrix dotRes = cu.dotGDwithWTrans(*gradients.back(),cuWeights);
        CuMatrix cuZ= cu.wrapMatrix(graphData.zMatrices[i - 1]);
        CuMatrix derivateRes = cu.activateDerivate(cuZ);
        gradients.push_back(cu.hadamardMul(dotRes, derivateRes));
    }

    std::reverse(gradients.begin(), gradients.end());

    // Compute weights updates.
    for (unsigned i = 0; i < gradients.size(); ++i){
        CuMatrix cuAct_=cu.wrapMatrix(graphData.actMatrices[i]);
        Matrix update=cu.dotActTranswithGD(cuAct_, *gradients[i], LEARNING_RATE).getMatrix();
        weightsUpdates.push_back(update);
    }

    for(auto g:gradients)
        delete g;

    return weightsUpdates;
}


/**
 *
 * Request the graph feature matrices data from dataserver.
 * 
 */ 
GraphData
ComputingServer::requestForwardMatrices(unsigned numLayers) {
    zmq::message_t confirm(5);    
    // Send pull request.
    zmq::message_t header(HEADER_SIZE);
    populateHeader((char *) header.data(), OP::PULL_BACKWARD, 0);
    GraphData graphData;

    // Receive z matrices chunks, from layer 1 -> last.
    for (size_t i = 1; i <= numLayers; ++i) {
        zmq::message_t respHeader;
        dataSocket.recv(&respHeader);

        unsigned rows = parse<unsigned>((char *) respHeader.data(), 2);
        unsigned cols = parse<unsigned>((char *) respHeader.data(), 3);
        zmq::message_t matxData(rows * cols * sizeof(FeatType));
        dataSocket.recv(&matxData);

        FeatType *matxBuffer = new FeatType[rows * cols];
        std::memcpy(matxBuffer, matxData.data(), matxData.size());

        graphData.zMatrices.push_back(Matrix(rows, cols, matxBuffer));
    }

    // Receive act matrices chunks, from layer 0 -> last.
    for (size_t i = 0; i <= numLayers; ++i) {
        zmq::message_t respHeader;
        dataSocket.recv(&respHeader);
        
        unsigned layerResp = parse<unsigned>((char *) respHeader.data(), 1);
       
       
        unsigned rows = parse<unsigned>((char *) respHeader.data(), 2);
        unsigned cols = parse<unsigned>((char *) respHeader.data(), 3);
        zmq::message_t matxData(rows * cols * sizeof(FeatType));
        dataSocket.recv(&matxData);

        FeatType *matxBuffer = new FeatType[rows * cols];
        std::memcpy(matxBuffer, matxData.data(), matxData.size());

        graphData.actMatrices.push_back(Matrix(rows, cols, matxBuffer));
    
    }

    // Receive target label matrix chunk.
    zmq::message_t respHeader;
    dataSocket.recv(&respHeader);

    unsigned layerResp = parse<unsigned>((char *) respHeader.data(), 1);
    
    unsigned rows = parse<unsigned>((char *) respHeader.data(), 2);
    unsigned cols = parse<unsigned>((char *) respHeader.data(), 3);
    zmq::message_t matxData(rows * cols * sizeof(FeatType));

    dataSocket.recv(&matxData);
    dataSocket.send(confirm);

    FeatType *matxBuffer = new FeatType[rows * cols];
    std::memcpy(matxBuffer, matxData.data(), matxData.size());

    graphData.targetMatrix = Matrix(rows, cols, matxBuffer);

    return graphData;
}

std::vector<Matrix> 
ComputingServer::requestWeightsMatrices(unsigned numLayers){
    // Send pull request.
    zmq::message_t header(HEADER_SIZE);
    populateHeader((char *) header.data(), OP::PULL_BACKWARD, 0);
    weightSocket.send(header);

    std::vector<Matrix> weightsData;

    // Receive weight matrices, from layer 2 -> last.
    for (size_t i = 2; i <= numLayers; ++i) {
        zmq::message_t respHeader;
        weightSocket.recv(&respHeader);
        unsigned layerResp = parse<unsigned>((char *) respHeader.data(), 1);
        if (layerResp == ERR_HEADER_FIELD) {    // Failed.
            std::cerr << "[ ERROR ] No corresponding weight matrix!" << std::endl;
            return weightsData;
        } else {                    // Get matrices data.
            unsigned rows = parse<unsigned>((char *) respHeader.data(), 2);
            unsigned cols = parse<unsigned>((char *) respHeader.data(), 3);
            zmq::message_t matxData(rows * cols * sizeof(FeatType));
            weightSocket.recv(&matxData);

            FeatType *matxBuffer = new FeatType[rows * cols];
            std::memcpy(matxBuffer, matxData.data(), matxData.size());

            weightsData.push_back(Matrix(rows, cols, matxBuffer));
        }
    }
    return weightsData;
}


void ComputingServer::processForward(zmq::message_t &header){
    unsigned layer = parse<unsigned>((char *) header.data(), 1);
    unsigned rows = parse<unsigned>((char *) header.data(), 2);
    unsigned cols = parse<unsigned>((char *) header.data(), 3);
    float split= parse<float>((char *) header.data(), 4);

    double t1=getTimer();
    Matrix weights;

    std::thread wThread=std::thread([&]{
        unsigned ipc_addr_len=strlen(ipc_addr);
        size_t identity_len = sizeof(unsigned) + ipc_addr_len;
        char identity[identity_len];
        memcpy(identity, (char *) &nodeId, sizeof(unsigned));
        memcpy(identity + sizeof(unsigned), ipc_addr, ipc_addr_len);

        std::cout << "< GPU SERVER FORWARD > Asking weightserver..." << std::endl;
        weightSocket.setsockopt(ZMQ_IDENTITY, identity, identity_len);
        char whost_port[50];
        sprintf(whost_port, "tcp://%s:%u", weightServerAddrs.at(nodeId%weightServerAddrs.size()), wPort);
        printf("connect to %s\n", whost_port);
        weightSocket.connect(whost_port);

        weights=requestWeightsMatrix(layer);
        printf("FORWARD Weight Fetch Time: %lf\n", getTimer()-t1);
        std::cout << "<  GPU SERVER FORWARD > Got data from weightserver." << std::endl;
    });
    double t3=getTimer();
    Matrix feats=requestFeatsMatrix(rows,cols);
    printf("FORWARD Features Fetch Time: %lf\n", getTimer()-t3);
    wThread.join();
    
    CuMatrix z = cu.dot(feats, weights);
    FeatType* z_buffer=new FeatType[z.getRows()*z.getCols()];
    memcpy(z_buffer,z.getData(),z.getDataSize());
    Matrix z_send(z.getRows(),z.getCols(),z_buffer);            
    cu.activate(z);//z data get activated ...
    printf("FORWARD Computation Time: %lf\n", getTimer()-t1);
    sendMatrices(z_send,z);

    
    if(split!=0)
        evaluateModel(z);

    delete[] (feats.getData());
    delete[] (z_buffer);
}

//Send multiplied matrix result back to dataserver.
void ComputingServer::sendMatrices(Matrix& zResult, Matrix& actResult) {
        zmq::message_t confirm;
        // // Send push header.
        zmq::message_t header(HEADER_SIZE);
        populateHeader((char *) header.data(),zResult.getRows(), zResult.getCols());
        dataSocket.send(header);
        dataSocket.recv(&confirm);
        // // Send zData and actData.
        zmq::message_t zData(zResult.getData(),zResult.getDataSize(),doNotFreeBuffer, NULL);
        dataSocket.send(zData);
        dataSocket.recv(&confirm);
        zmq::message_t actData(actResult.getData(),actResult.getDataSize(),doNotFreeBuffer, NULL);
        dataSocket.send(actData);
        dataSocket.recv(&confirm);
        dataSocket.send(confirm); 
}


void ComputingServer::loadWeightServers(std::vector<char *>& addresses, const std::string& wServersFile){
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

//For forward
Matrix
ComputingServer::requestFeatsMatrix(unsigned rows,unsigned cols) {
    zmq::message_t confirm(5);
    zmq::message_t aggreChunk;
    std::cout << "< GPU SERVER FORWARD  > Getting data from dataserver..." << std::endl;
    dataSocket.recv(&aggreChunk);
    std::cout << "< GPU SERVER FORWARD > Got data from dataserver." << std::endl;
    FeatType * feats_buffer=new FeatType[rows * cols];
    memcpy((char*)feats_buffer,(char*)aggreChunk.data(),rows * cols * sizeof(FeatType));
    Matrix m(rows, cols,feats_buffer);
    return m;
}

/**
 *
 * Request the input matrix data from weightserver.
 * 
 */
Matrix
ComputingServer::requestWeightsMatrix( unsigned layer) {
    
    // Send pull request.
    zmq::message_t header(HEADER_SIZE);
    populateHeader((char *) header.data(), OP::PULL_FORWARD, layer);
    weightSocket.send(header);

    // Listen on respond.
    zmq::message_t respHeader(HEADER_SIZE);
    weightSocket.recv(&respHeader);

    // Parse the respond.
    unsigned layerResp = parse<unsigned>((char *) respHeader.data(), 1);
    if ((int)layerResp == -1) {      // Failed.
        std::cerr << "[ ERROR ] No corresponding matrix chunk!" << std::endl;
        return Matrix();
    } else {                    // Get matrices data.
        unsigned rows = parse<unsigned>((char *) respHeader.data(), 2);
        unsigned cols = parse<unsigned>((char *) respHeader.data(), 3);
        zmq::message_t wData(rows * cols * sizeof(FeatType));
        weightSocket.recv(&wData);
        FeatType *wBuffer = new FeatType[rows*cols];
        memcpy((char*)wBuffer,(char*)wData.data(),rows * cols * sizeof(FeatType));
        Matrix m(rows, cols, wBuffer);
        return m;
    }
}

void 
ComputingServer::terminateWeightServers(){
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
    }
}

void
ComputingServer::sendShutdownMessage(zmq::socket_t& weightsocket) {
    zmq::message_t header(HEADER_SIZE);
    populateHeader((char *) header.data(), OP::TERM);
    weightsocket.send(header);
    
    // Set receive timeou 1s property on this weightsocket, in case that a weightserver is dying too quickly that it's
    // confirm message it not sent from buffer yet. Using timeout here because shutdown is not a big deal.
    weightsocket.setsockopt(ZMQ_RCVTIMEO, 1000);

    // Wait for termination confirmed reply.
    zmq::message_t confirm;
    weightsocket.recv(&confirm);
}
