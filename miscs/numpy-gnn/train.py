import numpy as np

from network import Network
from layers import Relu, Linear, Tanh, Aggregate
from utils import LOG_INFO
from loss import EuclideanLoss, SoftmaxCrossEntropyLoss
from solve_net import train_net, test_net
from load_data import load_data, load_weights


# As an example of 3 layers GCN
def example_GCN(name, adj, weights, layer_config):
    model = Network()
    model.add(Aggregate('A1', adj))
    model.add(Linear('W1', layer_config[0], layer_config[1], 'kaiming'))
    model.add(Relu('Relu1'))
    model.add(Aggregate('A2', adj))
    model.add(Linear('W2', layer_config[1], layer_config[1], 'kaiming'))
    model.add(Relu('Relu2'))
    model.add(Aggregate('A3', adj))
    model.add(Linear('W3', layer_config[1], layer_config[2], 'kaiming'))

    loss = SoftmaxCrossEntropyLoss(name='loss')

    print("Model "+name)
    for layer in model.layer_list:
        print(":\t" + repr(layer))
    print(':\t' + repr(loss))

    print('Forward Computation: ', model.str_forward('X'))
    print('Backward Computation:', model.str_backward('Z-Y'))
    print()
    model.str_update()
    print()

    return model, loss


def MLP(name, weights, layer_config):
    num_layer = len(layer_config)

    model = Network()
    for i in range(num_layer - 2):
        model.add(Linear('W{}'.format(i),
                         layer_config[i], layer_config[i + 1], 'kaiming'))
        model.add(Relu('Relu{}'.format(i)))

    model.add(Linear('W{}'.format(num_layer - 2),
                     layer_config[-2], layer_config[-1], 'kaiming'))

    loss = SoftmaxCrossEntropyLoss(name='loss')

    print("Model "+name)
    for layer in model.layer_list:
        print(":\t" + repr(layer))
    print(':\t' + repr(loss))
    print()

    print('Forward Computation: ', model.str_forward('X'))
    print('Backward Computation:', model.str_backward('Z-Y'))
    print()
    model.str_update()
    print()

    return model, loss


def GCN(name, adj, weights, layer_config):
    num_layer = len(layer_config)

    model = Network()
    for i in range(num_layer - 2):
        model.add(Aggregate('A{}'.format(i), adj))
        model.add(Linear('W{}'.format(i),
                         layer_config[i], layer_config[i + 1], 'kaiming'))
        model.add(Relu('Relu{}'.format(i)))

    model.add(Aggregate('A{}'.format(num_layer - 2), adj))
    model.add(Linear('W{}'.format(num_layer - 2),
                     layer_config[-2], layer_config[-1], 'kaiming'))

    loss = SoftmaxCrossEntropyLoss(name='loss')
    # loss = EuclideanLoss(name='loss')

    print("Model "+name)
    for layer in model.layer_list:
        print(":\t" + repr(layer))
    print(':\t' + repr(loss))

    print('Forward Computation: ', model.str_forward('X'))
    print('Backward Computation:', model.str_backward('Z-Y'))
    print()
    model.str_update()
    print()

    return model, loss


def train(adj, input_feats, target_labels, config, weights=None):
    num_vertices = adj.shape[0]
    label_kind = np.max(target_labels) + 1
    feat_dim = input_feats.shape[-1]
    layer_config = (feat_dim, config['hidden_dim'], label_kind)

    model, loss = GCN("GCN", adj, weights, layer_config)
    # model, loss = MLP("MLP", weights, layer_config)

    # Construct masks for training and testing
    train_size = int(num_vertices * config['train_portion'])
    train_mask = np.zeros(target_labels.shape, dtype=bool)
    train_mask[:train_size] = True
    np.random.shuffle(train_mask)
    test_mask = ~train_mask

    for epoch in range(config['max_epoch']):
        LOG_INFO('Training @ %d epoch...' % (epoch))
        train_net(model, loss, config, input_feats,
                  target_labels, train_mask, label_kind)

        if (epoch + 1) % config['test_epoch'] == 0:
            LOG_INFO('Testing @ %d epoch...' % (epoch))
            test_net(model, loss, input_feats,
                     target_labels, test_mask, label_kind)

        if (epoch + 1) % 50 == 0:
            config['learning_rate'] *= 0.5


def main():
    config = {
        'train_portion': 1.0,
        'hidden_dim': 16,
        'learning_rate': 0.01,
        'weight_decay': 5e-4,
        'momentum': 0.0,
        'max_epoch': 100,
        'test_epoch': 10,
        'check': False
    }

    A_hat, input_feats, target_labels = load_data(
        '/home/yifan/dataset/cora/', 'cora')

    train(A_hat, input_feats, target_labels, config)


if __name__ == "__main__":
    main()
