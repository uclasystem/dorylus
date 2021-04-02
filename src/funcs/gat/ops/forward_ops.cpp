#include "forward_ops.hpp"

Matrix
softmax(Matrix& mat) {
    FeatType* result = new FeatType[mat.getNumElemts()];

    for (unsigned r = 0; r < mat.getRows(); ++r) {
        unsigned length = mat.getCols();
        FeatType* vecSrc = mat.getData() + r * length;
        FeatType* vecDst = result + r * length;

        FeatType denom = 1e-20;
        FeatType maxEle = *(std::max_element(vecSrc, vecSrc + length));
        for (unsigned c = 0; c < length; ++c) {
            vecDst[c] = std::exp(vecSrc[c] - maxEle);
            denom += vecDst[c];
        }

        for (unsigned c = 0; c < length; ++c) {
            vecDst[c] /= denom;
        }
    }

    return Matrix(mat.getRows(), mat.getCols(), result);
}

Matrix
tanh(Matrix& mat) {
    FeatType *activationData = new FeatType[mat.getNumElemts()];
    FeatType *zData = mat.getData();

    for (unsigned i = 0; i < mat.getNumElemts(); ++i)
        activationData[i] = std::tanh(zData[i]);

    return Matrix(mat.getRows(), mat.getCols(), activationData);
}

Matrix
leakyReLU(Matrix& mat) {
    float alpha = .01;
    FeatType* result = new FeatType[mat.getNumElemts()];
    FeatType* matxData = mat.getData();

    for (unsigned i = 0; i < mat.getNumElemts(); ++i)
        result[i] = (matxData[i] > 0) ? matxData[i] : alpha * matxData[i];

    return Matrix(mat.getRows(), mat.getCols(), result);
}

unsigned
getMaxIndex(FeatType* row, unsigned length) {
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

unsigned
getLabelIndex(FeatType* row, unsigned length) {
    for (unsigned col = 0; col < length; ++col) {
        if (row[col] == 1)
            return col;
    }

    // Should never get here
    return -1;
}

unsigned
checkAccuracy(Matrix& predictions, Matrix& labels) {
    assert(predictions.getRows() == labels.getRows());
    assert(predictions.getCols() == labels.getCols());
    unsigned totalCorrect = 0;
    unsigned length = predictions.getCols();
    for (unsigned r = 0; r < predictions.getRows(); ++r) {
        unsigned maxIndex = getMaxIndex(predictions.get(r), length);

        if (labels.get(r, maxIndex) == 1.0)
            ++totalCorrect;
    }

    return totalCorrect;
}

float
checkLoss(Matrix& preds, Matrix& labels) {
    assert(preds.getRows() == labels.getRows());
    assert(preds.getCols() == labels.getCols());

    float totalLoss = 0;
	unsigned length = preds.getCols();
    for (unsigned r = 0; r < preds.getRows(); ++r) {
        unsigned labelIndex = getLabelIndex(labels.get(r), length);
        // loss = -log(class_prediction)
        float lossThisRow = -(std::log(preds.get(r, labelIndex)));
        totalLoss += lossThisRow;
    }

    return totalLoss;
}

Matrix
edgeMatMul(EdgeInfo& eInfo, Matrix& A, Matrix& B) {
    FeatType* result = new FeatType[eInfo.nChunkEdges * 1];

    FeatType* weightValues = B.getData();

//    for (unsigned u = 0; u < A.getCols(); ++u) {
//        std::cout << weightValues[u] << " ";
//    }
//    std::cout << std::endl;
//
//    for (unsigned vid = 0; vid < eInfo.numLvids; ++vid) {
//        FeatType* vidFeats = A.get(vid);
//        for (unsigned vInd = 0; vInd < A.getCols(); ++vInd) {
//            std::cout << vidFeats[vInd] << " ";
//        }
//        std::cout << std::endl;
//    }

    unsigned eIndex = 0;
    for (unsigned vid = 0; vid < eInfo.numLvids; ++vid) {
        FeatType* vidFeats = A.get(vid);
        for (unsigned eid = 0; eid < eInfo.edgePtrs[vid + 1] - eInfo.edgePtrs[vid];
             ++eid) {
            FeatType eValue = 0.0;
            for (unsigned v = 0; v < A.getCols(); ++v) {
                eValue += vidFeats[v] * weightValues[v];
            }
            result[eIndex++] = eValue;
        }
    }

    return Matrix(eInfo.nChunkEdges, 1, result);
}
