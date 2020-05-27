#include "CPU_comm.hpp"

#include <omp.h>
using namespace std;

CPUComm::CPUComm(Engine *engine_)
    : engine(engine_), nodeId(engine_->nodeId), totalLayers(engine_->numLayers),
    wServersFile(engine_->weightserverIPFile), wPort(engine_->weightserverPort),
    numNodes(engine_->numNodes), currLayer(0), savedNNTensors(engine_->savedNNTensors),
    msgService(wPort, nodeId)
{
    loadWeightServers(weightServerAddrs, wServersFile);
    msgService.setUpWeightSocket(
        weightServerAddrs.at(nodeId % weightServerAddrs.size()));
}

CPUComm::~CPUComm() {
    for (char *addr : weightServerAddrs) {
        free(addr);
    }
}

void CPUComm::NNCompute(Chunk &chunk) {
    c=chunk;
    currLayer = chunk.layer;

    if (chunk.dir == PROP_TYPE::FORWARD) {
        if (chunk.vertex) {
            // printLog(nodeId, "CPU FORWARD vtx NN started");
            vtxNNForward(currLayer, currLayer == (totalLayers - 1));
        } else {
            // printLog(nodeId, "CPU FORWARD edg NN started");
            edgNNForward(currLayer, currLayer == (totalLayers - 1));
        }
    }
    if (chunk.dir == PROP_TYPE::BACKWARD) {
        if (chunk.vertex) {
            // printLog(nodeId, "CPU BACKWARD vtx NN started");
            vtxNNBackward(currLayer);
        } else {
            // printLog(nodeId, "CPU BACKWARD edg NN started");
            edgNNBackward(currLayer);
        }
    }
    // printLog(nodeId, "CPU NN Done");
}

void CPUComm::vtxNNForward(unsigned layer, bool lastLayer) {
    Matrix feats;
    if (layer == 0) {
        feats = savedNNTensors[layer]["h"];
    } else {
        feats = savedNNTensors[layer - 1]["ah"];
    }
    Matrix weight = msgService.getWeightMatrix(layer);

    Matrix z = feats.dot(weight);

    memcpy(savedNNTensors[layer]["z"].getData(), z.getData(), z.getDataSize());
    deleteMatrix(z);
}

void CPUComm::vtxNNBackward(unsigned layer) {
    Matrix weight = msgService.getWeightMatrix(layer);
    Matrix grad = savedNNTensors[layer]["aTg"];
    Matrix h;
    if (layer == 0) {
        h = savedNNTensors[layer]["h"];
    } else {
        h = savedNNTensors[layer - 1]["ah"];
    }

    Matrix weightUpdates = h.dot(grad, true, false);
    msgService.sendWeightUpdate(weightUpdates, layer);
    cout<<"weightUpdates "<<weightUpdates.shape()<<endl;

    weightUpdates.free();

    if (layer != 0) {
        Matrix resultGrad = grad.dot(weight, false, true);
        memcpy(savedNNTensors[layer - 1]["grad"].getData(), resultGrad.getData(),
               resultGrad.getDataSize());
        resultGrad.free();
    }
}

void CPUComm::edgNNForward(unsigned layer, bool lastLayer) {
    Matrix a = msgService.getaMatrix(layer);
    Matrix z = savedNNTensors[layer]["z"];

    // expand and dot
    Matrix zaTensor = expandDot(z, a, engine->graph.forwardAdj);
    memcpy(savedNNTensors[layer]["az"].getData(), zaTensor.getData(), zaTensor.getDataSize());
    Matrix outputTensor = leakyRelu(zaTensor);
    zaTensor.free();

    memcpy(savedNNTensors[layer]["A"].getData(), outputTensor.getData(), outputTensor.getDataSize());
    outputTensor.free();
}

void CPUComm::edgNNBackward(unsigned layer) {
    Matrix a = msgService.getaMatrix(layer);

    Matrix gradTensor = savedNNTensors[layer]["grad"];
    Matrix zaTensor = savedNNTensors[layer]["az"];
    Matrix localZTensor = savedNNTensors[layer]["z"]; // serve as Z_dst, and part of Z_src
    // Matrix ghostZTensor = savedNNTensors[layer]["fg_z"]; // serve as part of Z_src
    // FeatType **fedge = engine->savedEdgeTensors[layer]["fedge"]; // This serves the purpose of Z_src and Z_dst
    // unsigned edgCnt = engine->graph.forwardAdj.nnz;
    // unsigned featDim = gradTensor.getCols();

    // gradient of LeakyRelu
    Matrix dLRelu = leakyReluBackward(zaTensor);
    // expand dP to (|E|, featDim) and element-wise multiply dLRelu
    // Shape of dAct is (|E|, featDim)
    Matrix dAct = expandHadamardMul(gradTensor, dLRelu, engine->graph.forwardAdj);
    dLRelu.free();

    if (layer != 0) {
        // Shape of dA: (|E|, 1), serve as gradient of each edge for backward agg
        Matrix dA = dAct.dot(a);
        memcpy(savedNNTensors[layer]["dA"].getData(), dA.getData(), dA.getDataSize());
        dA.free();
    }

    // reduce dAct(|E|, featDim) to (1, featDim)
    // Do this first to optmize computation and save memory
    Matrix dAct_reduce = reduce(dAct);
    dAct.free();
    // // Expand Z_src and Z_dst (both have shape (|V|, featDim)) to (|E|, featDim)
    // // And then do Z_dst^T \dot Z_src -> zz (featDim, featDim)
    // Matrix zz = expandMulZZ(fedge, edgCnt, featDim);
    Matrix zz = localZTensor.dot(localZTensor, true, false);
    // (1, featDim) \dot (featDim, featDim) -> (1, featDim), which is da's shape
    Matrix da = zz.dot(dAct_reduce, false, true);
    dAct_reduce.free();
    zz.free();
    msgService.sendaUpdate(da, layer);
    da.free();
}

void loadWeightServers(std::vector<char *> &addresses,
                       const std::string &wServersFile) {
    std::ifstream infile(wServersFile);
    if (!infile.good())
        printf("Cannot open weight server file: %s [Reason: %s]\n",
               wServersFile.c_str(), std::strerror(errno));

    assert(infile.good());

    std::string line;
    while (!infile.eof()) {
        std::getline(infile, line);
        boost::algorithm::trim(line);

        if (line.length() == 0) continue;

        char *addr = strdup(line.c_str());
        addresses.push_back(addr);
    }
}

Matrix expandDot(Matrix &m, Matrix &v, CSCMatrix<EdgeType> &forwardAdj) {
    FeatType *outputData = new FeatType[forwardAdj.nnz];
    Matrix outputTensor(forwardAdj.nnz, 1, outputData);
    memset(outputData, 0, outputTensor.getDataSize());

    unsigned vtcsCnt = m.getRows();
    unsigned featDim = m.getCols();
    FeatType *vPtr = v.getData();
#pragma omp parallel for
    for (unsigned lvid = 0; lvid < vtcsCnt; lvid++) {
        FeatType *mPtr = m.get(lvid);
        for (unsigned long long eid = forwardAdj.columnPtrs[lvid];
            eid < forwardAdj.columnPtrs[lvid + 1]; ++eid) {
            for (unsigned j = 0; j < featDim; ++j) {
                outputData[eid] += mPtr[j] * vPtr[j];
            }
        }
    }

    return outputTensor;
}

Matrix expandHadamardMul(Matrix &m, Matrix &v, CSCMatrix<EdgeType> &forwardAdj) {
    unsigned vtcsCnt = m.getRows();
    unsigned featDim = m.getCols();
    unsigned edgCnt = forwardAdj.nnz;

    FeatType *outputData = new FeatType[edgCnt * featDim];
    Matrix outputTensor(forwardAdj.nnz, featDim, outputData);
    memset(outputData, 0, outputTensor.getDataSize());

    FeatType *vPtr = v.getData();
#pragma omp parallel for
    for (unsigned lvid = 0; lvid < vtcsCnt; lvid++) {
        FeatType *mPtr = m.get(lvid);
        for (unsigned long long eid = forwardAdj.columnPtrs[lvid];
            eid < forwardAdj.columnPtrs[lvid + 1]; ++eid) {
            FeatType normFactor = vPtr[eid];
            for (unsigned j = 0; j < featDim; ++j) {
                outputData[eid * featDim + j] = mPtr[j] * normFactor;
            }
        }
    }

    return outputTensor;
}

Matrix expandMulZZ(FeatType **eFeats, unsigned edgCnt, unsigned featDim) {
    FeatType *zzData = new FeatType[featDim * featDim];
    Matrix zzTensor(featDim, featDim, zzData);
    memset(zzData, 0, zzTensor.getDataSize());

    FeatType **srcFeats = eFeats;
    FeatType **dstFeats = eFeats + edgCnt;

#pragma omp parallel for
    for (unsigned i = 0; i < featDim; i++) {
        for (unsigned eid = 0; eid < edgCnt; eid++) {
            for (unsigned j = 0; j < featDim; ++j) {
                zzData[i * featDim + j] += srcFeats[eid][i] * dstFeats[eid][j];
            }
        }
    }

    return zzTensor;
}

Matrix reduce(Matrix &mat) {
    unsigned edgCnt = mat.getRows();
    unsigned featDim = mat.getCols();

    FeatType *outputData = new FeatType[featDim];
    Matrix outputTensor(1, featDim, outputData);

    FeatType *mPtr = mat.getData();
#pragma omp parallel for
    for (unsigned i = 0; i < featDim; i++) {
        for (unsigned eid = 0; eid < edgCnt; eid++) {
            outputData[i] += mPtr[eid * featDim + i];
        }
    }

    return outputTensor;
}

Matrix leakyRelu(Matrix &mat) {
    FeatType alpha = 0.01;
    FeatType *activationData = new FeatType[mat.getNumElemts()];
    FeatType *inputData = mat.getData();

#pragma omp parallel for
    for (unsigned i = 0; i < mat.getNumElemts(); ++i) {
        activationData[i] = (inputData[i] > 0) ? inputData[i] : alpha * inputData[i];
    }

    return Matrix(mat.getRows(), mat.getCols(), activationData);
}

Matrix leakyReluBackward(Matrix &mat) {
    FeatType alpha = 0.01;
    FeatType *outputData = new FeatType[mat.getNumElemts()];
    FeatType *inputData = mat.getData();

#pragma omp parallel for
    for (unsigned i = 0; i < mat.getNumElemts(); ++i) {
        outputData[i] = (inputData[i] > 0) ? 1 : alpha;
    }

    return Matrix(mat.getRows(), mat.getCols(), outputData);
}

Matrix hadamardMul(Matrix &A, Matrix &B) {
    FeatType *result = new FeatType[A.getRows() * A.getCols()];

    FeatType *AData = A.getData();
    FeatType *BData = B.getData();

#pragma omp parallel for
    for (unsigned ui = 0; ui < A.getNumElemts(); ++ui) {
        result[ui] = AData[ui] * BData[ui];
    }

    return Matrix(A.getRows(), A.getCols(), result);
}

Matrix hadamardSub(Matrix &A, Matrix &B) {
    FeatType *result = new FeatType[A.getRows() * B.getCols()];

    FeatType *AData = A.getData();
    FeatType *BData = B.getData();

#pragma omp parallel for
    for (unsigned ui = 0; ui < A.getNumElemts(); ++ui) {
        result[ui] = AData[ui] - BData[ui];
    }

    return Matrix(A.getRows(), B.getCols(), result);
}

void CPUComm::getTrainStat(Matrix &preds, Matrix &labels, float &acc,
                           float &loss) {
    acc = 0.0;
    loss = 0.0;
    unsigned featDim = labels.getCols();
    for (unsigned i = 0; i < labels.getRows(); i++) {
        FeatType *currLabel = labels.getData() + i * labels.getCols();
        FeatType *currPred = preds.getData() + i * labels.getCols();
        acc += currLabel[argmax(currPred, currPred + featDim)];
        loss -= std::log(currPred[argmax(currLabel, currLabel + featDim)]);
    }
    acc /= labels.getRows();
    loss /= labels.getRows();
    printLog(nodeId, "batch loss %f, batch acc %f", loss, acc);
}

void deleteMatrix(Matrix &mat) {
    if (!mat.empty()) {
        delete[] mat.getData();
        mat = Matrix();
    }
}