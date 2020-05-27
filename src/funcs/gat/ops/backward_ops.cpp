#include "backward_ops.hpp"

Matrix
tanhDerivative(Matrix& mat) {
    FeatType *res = new FeatType[mat.getNumElemts()];
    FeatType *zData = mat.getData();

    for (unsigned i = 0; i < mat.getNumElemts(); ++i)
        res[i] = 1 - std::pow(std::tanh(zData[i]), 2);

    return Matrix(mat.getRows(), mat.getCols(), res);
}


Matrix
leakyReLUDerivative(Matrix& mat) {
    FeatType* result = new FeatType[mat.getNumElemts()];
    FeatType* inputData = mat.getData();

    for (unsigned ind = 0; ind < mat.getNumElemts(); ++ind) {
        result[ind] = inputData[ind] > 0 ? 1.0 : .01;
    }

    return Matrix(mat.getRows(), mat.getCols(), result);
}

Matrix expandDot(Matrix &m, Matrix &v, EdgeInfo &eInfo) {
    FeatType *outputData = new FeatType[eInfo.nChunkEdges];
    Matrix outputTensor(eInfo.nChunkEdges, 1, outputData);
    memset(outputData, 0, outputTensor.getDataSize());

    unsigned vtcsCnt = m.getRows();
    unsigned featDim = m.getCols();
    FeatType *vPtr = v.getData();

    unsigned edgIdx = 0;
    for (unsigned lvid = 0; lvid < vtcsCnt; lvid++) {
        FeatType *mPtr = m.get(lvid);
        for (unsigned long long eid = eInfo.edgePtrs[lvid];
             eid < eInfo.edgePtrs[lvid + 1]; ++eid) {
            for (unsigned j = 0; j < featDim; ++j) {
                outputData[eid] += mPtr[j] * vPtr[j];
            }
        }
        edgIdx++;
    }

    return outputTensor;
}

Matrix expandHadamardMul(Matrix &m, Matrix &v, EdgeInfo &eInfo) {
    unsigned vtcsCnt = m.getRows();
    unsigned featDim = m.getCols();
    unsigned edgCnt = eInfo.nChunkEdges;

    FeatType *outputData = new FeatType[edgCnt * featDim];
    Matrix outputTensor(edgCnt, featDim, outputData);
    memset(outputData, 0, outputTensor.getDataSize());

    FeatType *vPtr = v.getData();
    unsigned edgIdx = 0;
    for (unsigned lvid = 0; lvid < vtcsCnt; lvid++) {
        FeatType *mPtr = m.get(lvid);
        for (unsigned long long eid = eInfo.edgePtrs[lvid];
             eid < eInfo.edgePtrs[lvid + 1]; ++eid) {
            FeatType normFactor = vPtr[edgIdx];
            for (unsigned j = 0; j < featDim; ++j) {
                outputData[edgIdx * featDim + j] = mPtr[j] * normFactor;
            }
            edgIdx++;
        }
    }

    return outputTensor;
}

Matrix reduce(Matrix &mat) {
    unsigned edgCnt = mat.getRows();
    unsigned featDim = mat.getCols();

    FeatType *outputData = new FeatType[featDim];
    Matrix outputTensor(1, featDim, outputData);

    FeatType *mPtr = mat.getData();

    for (unsigned eid = 0; eid < edgCnt; eid++) {
        for (unsigned i = 0; i < featDim; i++) {
            outputData[i] += mPtr[eid * featDim + i];
        }
    }

    return outputTensor;
}


// for compatibility
Matrix expand(Matrix &mat, EdgeInfo &eInfo) {
    unsigned vtcsCnt = mat.getRows();
    unsigned featDim = mat.getCols();
    unsigned edgCnt = eInfo.nChunkEdges;

    FeatType *outputData = new FeatType[edgCnt * featDim];
    Matrix outputTensor(edgCnt, featDim, outputData);

    unsigned edgIdx = 0;
    for (unsigned lvid = 0; lvid < vtcsCnt; lvid++) {
        FeatType *mPtr = mat.get(lvid);
        for (unsigned long long eid = eInfo.edgePtrs[lvid];
             eid < eInfo.edgePtrs[lvid + 1]; ++eid) {
            for (unsigned j = 0; j < featDim; ++j) {
                outputData[edgIdx * featDim + j] = mPtr[j];
            }
            edgIdx++;
        }
    }

    return outputTensor;
}