import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops

export_operators_grad_module = tf.load_op_library('libexport_operators.so')

@ops.RegisterGradient("Aggregate")
def _aggregate_grad(op, grad):
    """
    The gradient for `aggregate` using the operation implemented in C++.

    :param op: `aggregate` `Operation` that we are differentiating, which we can use
        to find the inputs and outputs of the original op.
    :param grad: gradient with respect to the output of the `aggregate` op.
    :return: gradients with respect to the input of `aggregate`.
    """

    return export_operators_grad_module.aggregate_grad(grad, op.inputs[0])

@ops.RegisterGradient("ApplyVertex")
def _apply_vertex_grad(op, grad):
    return export_operators_grad_module.apply_vertex_grad(grad)

@ops.RegisterGradient("Scatter")
def _scatter_grad(op, grad):
    return export_operators_grad_module.scatter_grad(grad)