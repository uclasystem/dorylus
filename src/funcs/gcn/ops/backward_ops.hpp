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
// END COMPUTATION

void maskout(Matrix &preds, Matrix &labels);

#endif
