#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"

#include "../engine/engine.hpp"

using namespace tensorflow;


REGISTER_OP("Aggregate")
    .Input("vtcsTensor: float")
    .Output("aggVtcsTensor: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        shape_inference::ShapeHandle input_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &input_shape));

        shape_inference::DimensionHandle input_rows = c->Dim(input_shape, 0);
        shape_inference::DimensionHandle output_rows = c->Dim(input_shape, 1);
        shape_inference::DimensionHandle merged;
        TF_RETURN_IF_ERROR(c->Merge(input_rows, weight_cols, &merged));

        c->set_output(0, c->Matrix(output_rows, 1));
        return Status::OK();
    });

REGISTER_OP("AppleVertex")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        return Status::OK();
    });

REGISTER_OP("Scatter")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        return Status::OK();
    });

/// \brief Implementation of an inner product operation.
/// \param context
/// \author David Stutz
class AggregateOp : public OpKernel {
public:
    /// \brief Constructor.
    /// \param context
    explicit AggregateOp(OpKernelConstruction* context) : OpKernel(context) {
        // TODO: set engine pointer
        engine = NULL;
    }

    /// \brief Compute the inner product.
    /// \param context
    void Compute(OpKernelContext* context) override {

        // some checks to be sure ...
        DCHECK_EQ(1, context->num_inputs());

        // get the input tensor
        const Tensor& input = context->input(0);

        // check shapes of input and weights
        const TensorShape& input_shape = input.shape();

        // check input is a standing vector
        DCHECK_EQ(input_shape.dims(), 2);
        DCHECK_EQ(input_shape.dim_size(1), 1);

        // create output shape
        TensorShape output_shape;
        output_shape.AddDim(weights_shape.dim_size(0));
        output_shape.AddDim(1);

        // create output tensor
        Tensor* output = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));

        // get the corresponding Eigen tensors for data access
        auto input_tensor = input.matrix<float>();
        auto weights_tensor = weights.matrix<float>();
        auto output_tensor = output->matrix<float>();

        for (int i = 0; i < output->shape().dim_size(0); i++) {
            output_tensor(i, 0) = 0;
            for (int j = 0; j < weights.shape().dim_size(1); j++) {
                output_tensor(i, 0) += weights_tensor(i, j)*input_tensor(j, 0);
            }
        }
    }

    /// pointer to Engine class which holds all variables of graph engine
    Engine* engine;
};

class ApplyVertexOp : public OpKernel {
public:
    explicit ApplyVertexOp(OpKernelConstruction* context) : OpKernel(context) {
        // TODO: set engine pointer
    }

    void Compute(OpKernelContext* context) override {
        // ApplyVertex should have no input parameter
        DCHECK_EQ(0, context->num_inputs());
        // ApplyVertex just call the corresponding engine function
    }

    Engine* engine;
};

class ScatterOp : public OpKernel {
public:
    explicit ScatterOp(OpKernelConstruction* context) : OpKernel(context) {
        // TODO: set engine pointer
    }

    void Compute(OpKernelContext* context) override {
        // Scatter should have no input parameter
        DCHECK_EQ(0, context->num_inputs());
        // Scatter just send ghosts, and reset the environment for the next iteration
    }

    Engine* engine;
}


REGISTER_KERNEL_BUILDER(Name("Aggregate").Device(DEVICE_CPU), AggregateOp);
REGISTER_KERNEL_BUILDER(Name("ApplyVertex").Device(DEVICE_CPU), ApplyVertexOp);
REGISTER_KERNEL_BUILDER(Name("Scatter").Device(DEVICE_CPU), ScatterOp);
