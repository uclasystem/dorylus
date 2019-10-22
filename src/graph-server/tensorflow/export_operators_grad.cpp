#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

#include "../engine/engine.hpp"

using namespace tensorflow;

// the gradients are simply passed as additional arguments as
// they are available in the Python function for registering the gradient operation.
REGISTER_OP("AggregateGrad")
    .Input("grad: float32")
    .Input("input: float32")
    .Output("grad_input: float32")

REGISTER_OP("ApplyVertexGrad")
    .Input("grad: float32")

REGISTER_OP("ScatterGrad")
    .Input("grad: float32")

/// \brief Implementation of aggregate gradient operation.
/// Note that this operation is used in Python to register the gradient as
/// this is not possible in C*+ right now.
/// \param context
/// \author David Stutz
class AggregateGradOp : public OpKernel {
public:
    /// \brief Constructor.
    /// \param context
    explicit AggregateGradOp(OpKernelConstruction* context) : OpKernel(context) {
        engine = NULL;
    }

    /// \brief Compute the aggregate gradients.
    /// \param context
    void Compute(OpKernelContext* context) override {

        // output and grad is provided as input
        DCHECK_EQ(2, context->num_inputs());

        // get the gradient tensor
        const Tensor& grad = context->input(0);

        // get the original input tensor
        const Tensor& input = context->input(1);

        // create input shape (inferred from the additional attribute `n`)
        TensorShape input_shape = input.shape();

        DCHECK_EQ(input_shape.dim_size(0), weights_shape.dim_size(1));

        // create output tensors
        Tensor* grad_input = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, input_shape, &grad_input));

        // get the Eigen tensors for data access
        auto grad_tensor = grad.matrix<float>();
        auto input_tensor = input.matrix<float>();
        auto grad_input_tensor = grad_input->matrix<float>();

        // doign it manually for ismplicity
        for (int i = 0; i < weights_shape.dim_size(0); i++) {
            grad_input_tensor(i, 0) = 0;
            for (int j = 0; j < grad.shape().dim_size(0); j++) {
                grad_input_tensor(i, 0) += grad_tensor(j, 0)*weights_tensor(j, i);
            }
        }
    }

    Engine* engine;
};

class ApplyVertexGradOp : public OpKernel {
public:
    explicit ApplyVertexGradOp(OpKernelConstruction* context) : OpKernel(context) {
        engine = NULL;
    }

    void Compute(OpKernelContext* context) override {
        DCHECK_EQ(1, context->num_inputs());
    }

    Engine* engine;
};

class ScatterGradOp : public OpKernel {
public:
    explicit ScatterGradOp(OpKernelConstruction* context) : OpKernel(context) {
        engine = NULL;
    }

    void Compute(OpKernelContext* context) override {
        DCHECK_EQ(1, context->num_inputs());
    }

    Engine* engine;
};


REGISTER_KERNEL_BUILDER(Name("AggregateGrad").Device(DEVICE_CPU), AggregateGradOp);
REGISTER_KERNEL_BUILDER(Name("ApplyVertexGrad").Device(DEVICE_CPU), ApplyVertexGradOp);
REGISTER_KERNEL_BUILDER(Name("ScatterGrad").Device(DEVICE_CPU), ScatterGradOp);
