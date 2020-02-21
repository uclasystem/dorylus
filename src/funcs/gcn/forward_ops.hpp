#ifndef __FWD_OPS_HPP__
#define __FWD_OPS_HPP__

// COMPUTATION
/**
 *
 * Apply softmax to all rows of the input matrix (Currently overwrites the
 * input matrix data)
 *
 */
static Matrix softmax(Matrix& mat);

/**
 *
 * Apply softmax to all rows of the input matrix (Currently overwrites the
 * input matrix data)
 *
 */
static Matrix tanh(Matrix& mat);
// END COMPUTATION

// LOSS/ACCURACY
static unsigned getMaxIndex(FeatType* row, unsigned length);
static unsigned getLabelIndex(FeatType* row, unsigned length);
static unsigned checkAccuracy(Matrix& predictions, Matrix& labels);
static float checkLoss(Matrix& preds, Matrix& labels);
// END LOSS/ACCURACY

#endif
