#ifndef __FWD_OPS_HPP__
#define __FWD_OPS_HPP__

#include <algorithm>

#include "../../../common/matrix.hpp"
#include "../../../common/utils.hpp"

// COMPUTATION
/**
 *
 * Apply softmax to all rows of the input matrix (Currently overwrites the
 * input matrix data)
 *
 */
Matrix softmax(Matrix& mat);

/**
 *
 * Apply softmax to all rows of the input matrix (Currently overwrites the
 * input matrix data)
 *
 */
Matrix tanh(Matrix& mat);
// END COMPUTATION

// LOSS/ACCURACY
unsigned getMaxIndex(FeatType* row, unsigned length);
unsigned getLabelIndex(FeatType* row, unsigned length);
unsigned checkAccuracy(Matrix& predictions, Matrix& labels);
float checkLoss(Matrix& preds, Matrix& labels);
// END LOSS/ACCURACY

#endif
