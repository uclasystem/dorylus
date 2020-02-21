#ifndef __NTWK_OPS_HPP__
#define __NTWK_OPS_HPP__

static Matrix requestTensor(zmq::socket_t& socket, OP op, unsigned partId,
  TYPE type = TYPE::AH, unsigned layer = 0);

/**
 *
 * Send multiplied matrix result back to dataserver.
 *
 */
static void sendMatrices(Matrix& zResult, Matrix& actResult,
  zmq::socket_t& socket, unsigned id);

/**
 *
 * Send matrix back to a server.
 *
 */
static void sendMatrix(Matrix& matrix, zmq::socket_t& socket, unsigned id);


#endif
