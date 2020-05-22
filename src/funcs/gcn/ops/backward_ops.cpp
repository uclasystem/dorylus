#include "backward_ops.hpp"

Matrix
tanhDerivative(Matrix& mat) {
    FeatType *res = new FeatType[mat.getNumElemts()];
    FeatType *zData = mat.getData();

    for (unsigned i = 0; i < mat.getNumElemts(); ++i)
        res[i] = 1 - std::pow(std::tanh(zData[i]), 2);

    return Matrix(mat.getRows(), mat.getCols(), res);
}

void maskout(Matrix &preds, Matrix &labels) {
    unsigned end = labels.getRows();
    unsigned stt = (unsigned)(end * TRAIN_PORTION);

    FeatType *predStt = preds.get(stt);
    FeatType *labelStt = labels.get(stt);
    memcpy(predStt, labelStt, sizeof(FeatType) * (end - stt));
}