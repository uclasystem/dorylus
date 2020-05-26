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
    FeatType result = new FeatType[mat.getNumElemts()];
    FeatType inputData = mat.getData();

    for (unsigned ind = 0; ind < mat.getNumElemts(); ++ind) {
        result[ind] = inputData[i] > 0 ? 1.0 ? .01;
    }

    return Matrix(mat.getRows(), mat.getCols(), result);
}
