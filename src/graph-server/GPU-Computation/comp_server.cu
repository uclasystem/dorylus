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

ComputingServer::ComputingServer(GPUComm* gpu_comm){
    gpuComm=gpu_comm;
    nodeId=gpu_comm->nodeId;
    msgService=MessageService(gpu_comm->wPort,nodeId);
    loadWeightServers(weightServerAddrs,gpu_comm->wServersFile);
    msgService.setUpWeightSocket(weightServerAddrs.at(nodeId%weightServerAddrs.size()));
}

//Start listening to main thread
void ComputingServer::terminate(){   
    msgService.terminateWeightServers(weightServerAddrs);
}

void ComputingServer::processForward(unsigned layer, bool lastLayer){
    
    Matrix feats=gpuComm->actMatrix;
    Matrix weights=msgService.requestWeightsMatrix(layer);
    FeatType* z_data=gpuComm->zData;
    FeatType* act_z=gpuComm->actData;
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
    delete[] z.getData();
    delete[] weights.getData();
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

//     msgService.sendFourBytes((char*)&totalCorrect);
//     msgService.sendFourBytes((char*)&lossThisPart);
// }

void ComputingServer::processBackward(unsigned layer, bool lastLayer){

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

    if (lastLayer) {
        std::cout << "< BACKWARD > Computing gradient from loss" << std::endl;
        gradLoss(layer);
    } else {
        std::cout << "< BACKWARD > Computing gradient for this layer" << std::endl;
        gradLayer(layer);
    }
}

void
ComputingServer::gradLayer(unsigned layer) {

    std::cout << "< BACKWARD > Requesting gradient from graph server" << std::endl;
    Matrix grad = gpuComm->oldGradMatrix;
    CuMatrix cuGrad=cu.wrapMatrix(grad);
    std::cout << "< BACKWARD > Requesting Z values" << std::endl;
    Matrix z = gpuComm->savedTensors[layer][TYPE::Z - 1];
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
    resultGrad.setData(gpuComm->newGradMatrix.getData());
    resultGrad.updateMatrixFromGPU();

    std::cout << "< BACKWARD > Requesting AH" << std::endl;
    Matrix ah = gpuComm->savedTensors[layer][TYPE::AH - 1];
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
    Matrix predictions = gpuComm->savedTensors[layer][TYPE::ACT - 1];
    Matrix labels = gpuComm->targetMatrix;

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
    interGrad.setData(gpuComm->newGradMatrix.getData());
    interGrad.updateMatrixFromGPU();

    // AH^T * d_out
    std::cout << "< BACKWARD > Requesting AH" << std::endl;
    Matrix ah = gpuComm->savedTensors[layer][TYPE::AH - 1];
    std::cout << "< BACKWARD > Computing weight updates" << std::endl;
    CuMatrix cuAh=cu.wrapMatrix(ah);
    CuMatrix cuWeightUpdates = cuAh.dot(d_output, true, false, LEARNING_RATE);
    Matrix weightUpdates=cuWeightUpdates.getMatrix();

    std::cout << "< BACKWARD > Sending weight updates" << std::endl;
    msgService.sendWeightUpdate(weightUpdates,layer);
}




