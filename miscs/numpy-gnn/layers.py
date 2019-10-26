import numpy as np


class Layer(object):
    def __init__(self, name, trainable=False):
        self.name = name
        self.trainable = trainable
        self._saved_tensor = None

    def forward(self, input_feat):
        pass

    def backward(self, grad_output):
        pass

    def update(self, config):
        pass

    def _saved_for_backward(self, tensor):
        self._saved_tensor = tensor

    # for visualize auto-grad formulas
    def str_forward(self, str_input):
        return 'Unknown({})'.format(str_input)

    def str_backward(self, str_grad):
        return 'Unknown({})'.format(str_grad)

    def str_update(self, config):
        pass

    def __repr__(self):
        return 'IR: Unknown Layer'


class Relu(Layer):
    def __init__(self, name):
        super(Relu, self).__init__(name)

    def forward(self, input_feat):
        self._saved_for_backward(input_feat)
        return np.maximum(0, input_feat)

    def backward(self, grad_output):
        input_feat = self._saved_tensor
        return grad_output * (input_feat > 0)

    def str_forward(self, str_input):
        return 'Relu({})'.format(str_input)

    def str_backward(self, str_grad):
        return 'Relu\'({})'.format(str_grad)

    def __repr__(self):
        return 'IR: Relu(input)'


class Tanh(Layer):
    def __init__(self, name):
        super(Tanh, self).__init__(name)

    def forward(self, input_feat):
        output = np.tanh(input_feat)
        self._saved_for_backward(output)
        return output

    def backward(self, grad_output):
        output = self._saved_tensor
        return grad_output * (1 - np.multiply(output, output))

    def str_forward(self, str_input):
        return 'Tanh({})'.format(str_input)

    def str_backward(self, str_grad):
        return 'Tanh\'({})'.format(str_grad)

    def __repr__(self):
        return 'IR: Tanh(input)'


class Sigmoid(Layer):
    def __init__(self, name):
        super(Sigmoid, self).__init__(name)

    def forward(self, input_feat):
        output = 1 / (1 + np.exp(-input_feat))
        self._saved_for_backward(output)
        return output

    def backward(self, grad_output):
        output = self._saved_tensor
        return grad_output * output * (1 - output)

    def str_forward(self, str_input):
        return 'Tanh({})'.format(str_input)

    def str_backward(self, str_grad):
        return 'Tanh\'({})'.format(str_grad)

    def __repr__(self):
        return 'IR: Sigmoid(input)'


class Linear(Layer):
    ''' full connection layer '''

    def __init__(self, name, in_num, out_num, init_method, init_std=0.5):
        super(Linear, self).__init__(name, trainable=True)
        # disabled bias to suit our GCN model
        self.bias = False
        self.random_noise = True

        self.in_num = in_num
        self.out_num = out_num

        if init_method == 'xavier':
            # init_range = np.sqrt(6.0/(in_num + out_num))
            # self.W = np.random.uniform(
            #     size=(in_num, out_num), low=-init_range, high=init_range)
            # self.W = np.random.uniform(-np.sqrt(1./out_num),
            #                            np.sqrt(1./out_num), (in_num, out_num))
            self.W = np.random.randn(in_num, out_num) / np.sqrt(in_num)
        elif init_method == 'kaiming':
            self.W = np.random.randn(in_num, out_num) / np.sqrt(in_num / 2)
        else:  # uniform
            self.W = np.random.randn(in_num, out_num) * init_std
        self.grad_W = np.zeros((in_num, out_num))
        self.diff_W = np.zeros((in_num, out_num))

        if self.bias:
            self.b = np.zeros(out_num)
            self.grad_b = np.zeros(out_num)
            self.diff_b = np.zeros(out_num)

    # set Weights manually for correctness checking
    def set_W(self, weight):
        if (self.W.shape != weight.shape):
            print("input weight matrix doesn't match the layer setting")
        else:
            self.W = weight

        return self

    def get_W(self):
        return self.W

    def forward(self, input_feat):
        self._saved_for_backward(input_feat)
        output = input_feat @ self.W
        if self.bias:
            output += self.b
        return output

    def backward(self, grad_output):
        input_feat = self._saved_tensor
        self.grad_W = input_feat.T @ grad_output
        if self.bias:
            self.grad_b = np.sum(grad_output, axis=0)
        return grad_output @ self.W.T

    def update(self, config):
        mm = config['momentum']
        lr = config['learning_rate']
        wd = config['weight_decay']

        if (config['check']):
            self.diff_W = lr * self.grad_W
            self.W = self.W - self.diff_W
            print(self.name + " Weight Grad Agg: {}".format(np.abs(self.diff_W).sum()) + " Max abs element {}".format(np.max(np.abs(self.diff_W))))
            return

        self.diff_W = mm * self.diff_W + (self.grad_W + wd * self.W)
        self.W = self.W - lr * self.diff_W

        # Random noise to jump out the local optimal.
        if self.random_noise:
            self.W += np.random.normal(0, 0.01, (self.in_num, self.out_num))

        if self.bias:
            self.diff_b = mm * self.diff_b + (self.grad_b + wd * self.b)
            self.b = self.b - lr * self.diff_b

    def str_forward(self, str_input):
        self.str_input = str_input
        return '({}) @ {}'.format(str_input, self.name)

    def str_backward(self, str_grad):
        self.str_grad = '({}).T @ ({})'.format(self.str_input, str_grad)
        return '({}) @ {}.T'.format(str_grad, self.name)

    def str_update(self):
        return 'Update Layer {}: {}'.format(self.name, self.str_grad)

    def __repr__(self):
        # self.name[1:] is a dirty way to get layer number
        return 'IR: input @ {} + b{}'.format(self.name, self.name[1:]) if self.bias else 'IR: input @ {}'.format(self.name)


class Aggregate(Layer):
    ''' Aggragate layer to get all neighbors' features '''

    def __init__(self, name, adj):
        super(Aggregate, self).__init__(name)
        self.adj = adj

    def forward(self, input_feat):
        return self.adj @ input_feat

    def backward(self, grad_output):
        return self.adj.T @ grad_output

    def str_forward(self, str_input):
        return 'A @ ({})'.format(str_input)

    def str_backward(self, str_grad):
        return 'A.T @ ({})'.format(str_grad)

    def __repr__(self):
        return 'IR: A @ input'


class Reshape(Layer):
    def __init__(self, name, new_shape):
        super(Reshape, self).__init__(name)
        self.new_shape = new_shape

    def forward(self, input_feat):
        self._saved_for_backward(input_feat)
        return input_feat.reshape(*self.new_shape)

    def backward(self, grad_output):
        input_feat = self._saved_tensor
        return grad_output.reshape(*input_feat.shape)
