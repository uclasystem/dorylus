#ifndef __BKWD_OPS_HPP__
#define __BKWD_OPS_HPP__

#include "../../../common/matrix.hpp"
#include "../../../common/utils.hpp"

// COMPUTATION
/**
 *
 * Apply derivative of the activation function on a matrix.
 *
 */
Matrix tanhDerivative(Matrix& mat);

Matrix leakyReLUDerivative(Matrix& mat);

Matrix expandDot(Matrix &m, Matrix &v, EdgeInfo &eInfo);

Matrix expandHadamardMul(Matrix &m, Matrix &v, EdgeInfo &eInfo);

Matrix reduce(Matrix &mat);

// END COMPUTATION

Matrix expand(Matrix &mat, EdgeInfo &eInfo);

#endif
