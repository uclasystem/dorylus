#include "comp_server.cuh"

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
ComputingServer::ComputingServer():cu(ComputingUnit::getInstance()){};

ComputingServer::ComputingServer(GPUComm* gpu_comm):cu(ComputingUnit::getInstance()){
    gpuComm=gpu_comm;
    totalLayers=gpu_comm->totalLayers;
    nodeId=gpu_comm->nodeId;
    msgService=MessageService(gpu_comm->wPort,nodeId);
    loadWeightServers(weightServerAddrs,gpu_comm->wServersFile);
    msgService.setUpWeightSocket(weightServerAddrs.at(nodeId%weightServerAddrs.size()));

    //send INFO to weight server
    unsigned numNodes=gpuComm->numNodes;
    if(nodeId<weightServerAddrs.size()){
        unsigned count = 0; 
        for (size_t i=0;i<numNodes;++i){
            if(i%weightServerAddrs.size()==nodeId)
                count+=1;
        }
        msgService.sendInfoMessage(count);
    }
    msgService.prefetchWeightsMatrix(totalLayers);
}

//Start listening to main thread
void ComputingServer::terminate(){   
    msgService.terminateWeightServers(weightServerAddrs);
}

void ComputingServer::processForward(unsigned layer, bool lastLayer){    
    Matrix feats=gpuComm->actMatrix;
    FeatType* z_data=gpuComm->zData;
    FeatType* act_z=gpuComm->actData;

    auto t1 = std::chrono::high_resolution_clock::now();
    Matrix weight=msgService.getWeightMatrix(layer);
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "Forward Network "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count()
              << " milliseconds\n";
    CuMatrix z = cu.dot(feats, weight);
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
    delete[] z.getData();
    CuMatrix::freeGPU();
    auto t3 = std::chrono::high_resolution_clock::now();
    std::cout << "Forward Compute "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t3-t2).count()
              << " milliseconds\n";
    
}


// void ComputingServer::evaluateModel(Matrix& activations){
//     CuMatrix labels = cu.wrapMatrix(msgService.requestMatrix());
//     CuMatrix cuAct =cu.wrapMatrix(activations);
//     CuMatrix cuPredictions = cu.softmaxRows(cuAct);

//     // Check if the label with the highest probability after softmax is equal to the
//     // target label
//     unsigned totalCorrect = cu.checkAccuracy(cuPredictions, labels);

//     // Sum the individual losses of each vertex for this validation partition
//     float lossThisPart = cu.checkLoss(cuPredictions, labels);
// }

void ComputingServer::processBackward(unsigned layer, bool lastLayer){
    if (lastLayer) {
        gradLoss(layer);
    } else {
        gradLayer(layer);
        if(layer==0)
            msgService.prefetchWeightsMatrix(totalLayers);; //***CAUTION only works for one weight server
    }
    CuMatrix::freeGPU();
}

void
ComputingServer::gradLayer(unsigned layer) {
    auto t3 = std::chrono::high_resolution_clock::now();
    Matrix grad = gpuComm->oldGradMatrix;
    CuMatrix cuGrad=cu.wrapMatrix(grad);
    Matrix z = gpuComm->savedTensors[layer][TYPE::Z - 1];
    CuMatrix cuZ = cu.wrapMatrix(z);
    CuMatrix interGrad=cu.activateBackward(cuZ,cuGrad);

    Matrix weight=msgService.getWeightMatrix(layer);
    CuMatrix cuWeights=cu.wrapMatrix(weight);
    CuMatrix resultGrad = interGrad.dot(cuWeights, false, true);
    resultGrad.setData(gpuComm->newGradMatrix.getData());
    resultGrad.updateMatrixFromGPU();
    Matrix ah = gpuComm->savedTensors[layer][TYPE::AH - 1];
    CuMatrix cuAh=cu.wrapMatrix(ah);
    CuMatrix cuWeightUpdates = cuAh.dot(interGrad, true, false);
    Matrix weightUpdates=cuWeightUpdates.getMatrix();
    
    // printf("interGrad[0] %f\n", (interGrad.getMatrix().getData()[0]));
    // printf("Ah[0] %f\n",ah.getData()[0]);
    // printf("Weight Update[0] %f\n", weightUpdates.getData()[0]);
    
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "Backward Compute "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t2-t3).count()
              << " milliseconds\n";

    msgService.sendWeightUpdate(weightUpdates,layer); 
}


void
ComputingServer::gradLoss(unsigned layer) {
    auto t3 = std::chrono::high_resolution_clock::now();
    Matrix predictions = gpuComm->savedTensors[layer][TYPE::ACT - 1];
    Matrix labels = gpuComm->targetMatrix;

    CuMatrix cuPredictions=cu.wrapMatrix(predictions);
    CuMatrix cuLabels=cu.wrapMatrix(labels);
    CuMatrix d_output = cu.hadamardSub(cuPredictions, cuLabels);

    Matrix weight=msgService.getWeightMatrix(layer);
    CuMatrix cuWeights=cu.wrapMatrix(weight);
    CuMatrix interGrad = d_output.dot(cuWeights, false, true);
    interGrad.setData(gpuComm->newGradMatrix.getData());
    interGrad.updateMatrixFromGPU();

    Matrix ah = gpuComm->savedTensors[layer][TYPE::AH - 1];
    CuMatrix cuAh=cu.wrapMatrix(ah);
    CuMatrix cuWeightUpdates = cuAh.dot(d_output, true, false);
    Matrix weightUpdates=cuWeightUpdates.getMatrix();
    
    // printf("d_output[0] %f\n", (d_output.getMatrix().getData()[0]));
    // printf("Ah[0] %f\n",ah.getData()[0]);
    // printf("Weight Update‘[0] %f\n", weightUpdates.getData()[0]);

    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "Backward(Loss) Compute "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t2-t3).count()
              << " milliseconds\n";
              
    msgService.sendWeightUpdate(weightUpdates,layer); 
}




