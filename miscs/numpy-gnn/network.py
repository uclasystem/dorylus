class Network(object):
    ''' Iterate all layers in a model for forward computation.
        Iterate inversely for backward gradient propagation.
        Update all gradient updates finally.
    '''
    def __init__(self):
        self.layer_list = []
        self.params = []
        self.num_layers = 0

    def add(self, layer):
        self.layer_list.append(layer)
        self.num_layers += 1

        return layer

    def forward(self, input):
        output = input
        for i in range(self.num_layers):
            output = self.layer_list[i].forward(output)

        return output

    def backward(self, grad_output):
        grad_input = grad_output
        for i in range(self.num_layers - 1, -1, -1):
            grad_input = self.layer_list[i].backward(grad_input)

    def update(self, config):
        for i in range(self.num_layers):
            if self.layer_list[i].trainable:
                self.layer_list[i].update(config)

    def str_forward(self, str_input):
        str_output = str_input
        for i in range(self.num_layers):
            str_output = self.layer_list[i].str_forward(str_output)

        return str_output

    def str_backward(self, str_outgrad):
        str_ingrad = str_outgrad
        for i in range(self.num_layers - 1, -1, -1):
            str_ingrad = self.layer_list[i].str_backward(str_ingrad)

        return str_ingrad

    def str_update(self):
        for i in range(self.num_layers):
            if self.layer_list[i].trainable:
                print(self.layer_list[i].str_update())