#ifndef __OPERATORS_HPP__
#define __OPERATORS_HPP__

#include <algorithm>
#include <chrono>
#include <ratio>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <cmath>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <cblas.h>
#include <zmq.hpp>
#include <aws/lambda-runtime/runtime.h>

#include "../src/common/matrix.hpp"
#include "../src/common/utils.hpp"


struct Operator {
    Operator(std::string _name = "Unknown OP", bool _trainable = false):
        name(_name), trainable(_trainable) {}

    virtual Matrix& forward(Matrix &inputTensor);
    virtual Matrix& backward(Matrix &gradTensor);
    virtual void update();
    virtual void _savedForBackward(Matrix &tensor) {
        _saved_tensor = tensor;
    }

    std::string name;
    bool trainable;
    Matrix &_saved_tensor;
};

struct Tanh: public Operator {
    Tanh(std::string _name = "Anonymous Tanh", bool _trainable = false) : Operator(_name, _trainable) {}

    Matrix& forward(Matrix &inputTensor) {
        FeatType *outputData = new FeatType[inputTensor.getNumElemts()];
        FeatType *inputData = inputTensor.getData();

        for (unsigned i = 0; i < inputTensor.getNumElemts(); ++i)
            outputData[i] = std::tanh(inputData[i]);

        return Matrix(inputTensor.getRows(), inputTensor.getCols(), outputData);
    }

    Matrix& backward(Matrix &gradTensor) {
        FeatType *outputData = new FeatType[_saved_tensor.getNumElemts()];
        Matrix outputTensor = Matrix(gradTensor.getRows(), gradTensor.getCols(), outputData);

        for (unsigned i = 0; i < _saved_tensor.getNumElemts(); ++i) {
            outputData[i] = 1 - std::pow(std::tanh(_saved_tensor[i]), 2);
        }

        return gradTensor * outputTensor;
    }
};

struct Linear: public Operator {
    Linear(std::string _name = "Anonymous Linear", bool _trainable = false): Operator(_name, _trainable) {}
    Matrix& forward(Matrix &inputTensor) {
        // TODO: save the inputTensor for backprop, that is to say, send the input tensor back to graph servers
        return inputTensor.dot(W);
    }

    Matrix& backward(Matrix &gradTensor) {
        // TODO: get the saved_tensor some time before calling backward here. that is to say, get the input tensor from graph severs
        Matrix &inputTensor = _saved_tensor;
        gradW = inputTensor.dot(gradTensor, true, false, 1.0);

        return gradTensor.dot(W, false, true, 1.0);
    }

    // TODO: impl the update func here. Basically calc the diffW based on gradW refer to numpy-gnn's impl.

    Matrix &W, &b;
    Matrix &gradW, &gradb;
    unsigned inDim, outDim;
};


struct Softmax {
    Matrix& foward(Matrix &inputTensor, Matrix &target) {
        FeatType *inputData = inputTensor.getData();
        unsigned numElemts = inputTensor.getNumElemts();
        unsigned rows = inputTensor.getRows();
        unsigned cols = inputTensor.getCols();

        FeatType *outputData = new FeatType[numElemts];
        Matrix &outputTensor = Matrix(rows, cols, outputData);

        for (unsigned i = 0; i < rows; ++i) {
            FeatType *inputRow = inputData + i * cols;
            FeatType *outputRow = outputData + i * cols;
            FeatType denom = 1e-20;
            FeatType maxEle = std::max_element(inputRow, inputRow + cols);
            for (unsigned j = 0; j < cols; ++j) {
                outputRow[j] = std::exp(inputRow[j] - maxEle);
                denom += outputRow[j];
            }
            for (unsigned j = 0; j < cols; ++j) {
                outputRow[j] /= denom;
            }
        }

        return outputTensor;
    }

    Matrix& backward(Matrix &inputTensor, Matrix &target) {
        FeatType *inputData = inputTensor.getData();
        unsigned numElemts = inputTensor.getNumElemts();
        unsigned rows = inputTensor.getRows();
        unsigned cols = inputTensor.getCols();

        FeatType *outputData = new FeatType[numElemts];
        Matrix &outputTensor = Matrix(rows, cols, outputData);

        for (unsigned i = 0; i < rows; ++i) {
            FeatType *inputRow = inputData + i * cols;
            FeatType *outputRow = outputData + i * cols;
            FeatType denom = 1e-20;
            FeatType maxEle = std::max_element(inputRow, inputRow + cols);
            for (unsigned j = 0; j < cols; ++j) {
                outputRow[j] = std::exp(inputRow[j] - maxEle);
                denom += outputRow[j];
            }
            for (unsigned j = 0; j < cols; ++j) {
                outputRow[j] /= denom;
            }
        }

        outputTensor -= target;
        return outputTensor;
    }
};


#endif // __OPERATORS_HPP__