from __future__ import division
import numpy as np


class EuclideanLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        return 0.5 * np.mean(np.sum(np.square(input - target), axis=1))

    def backward(self, input, target):
        return (input - target) / len(input)

    def str_backward(self):
        return '(Z - Y) / len(Z)'

    def __repr__(self):
        return 'IR: MSE_loss(input)'


class SoftmaxCrossEntropyLoss(object):
    def __init__(self, name):
        self.name = name

    # def forward(self, input, target):
    #     exp_input = np.exp(input - np.max(input, axis=1, keepdims=True))
    #     prob = exp_input / (np.sum(exp_input, axis=1, keepdims=True) + 1e-20)
    #     return np.mean(np.sum(-target * np.log(prob + 1e-20), axis=1))

    # def backward(self, input, target):
    #     exp_input = np.exp(input - np.max(input, axis=1, keepdims=True))
    #     prob = exp_input / (np.sum(exp_input, axis=1, keepdims=True) + 1e-20)
    #     return (prob - target)

    # for correctness check
    def forward(self, input, target):
        exp_input = np.exp(input)
        prob = exp_input / np.sum(exp_input, axis=1, keepdims=True)
        return np.mean(np.sum(-target * np.log(prob), axis=1))

    def backward(self, input, target):
        exp_input = np.exp(input)
        prob = exp_input / (np.sum(exp_input, axis=1, keepdims=True))
        return (prob - target)

    def str_backward(self):
        return '(Z - Y)'

    def __repr__(self):
        return 'IR: log_softmax_loss(input)'
