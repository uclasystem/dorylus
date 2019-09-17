from utils import LOG_INFO, onehot_encoding, calculate_acc
import numpy as np


def train_net(model, loss, config, input_feats, labels, train_mask, label_kind):
    target = onehot_encoding(labels, label_kind)

    # forward net
    output = model.forward(input_feats)
    # set mask
    output[~train_mask] = target[~train_mask]
    # calculate loss
    loss_value = loss.forward(output, target)
    # generate gradient w.r.t loss
    grad = loss.backward(output, target)
    # backward gradient
    model.backward(grad)
    # update layers' weights
    model.update(config)

    acc_value = calculate_acc(output, labels, np.sum(train_mask))

    msg = '  Training batch loss %.4f, batch acc %.4f' % (
        loss_value, acc_value)
    LOG_INFO(msg)


def test_net(model, loss, input_feats, labels, test_mask, label_kind):
    target = onehot_encoding(labels, label_kind)
    output = model.forward(input_feats)

    # set mask
    output[~test_mask] = target[~test_mask]
    loss_value = loss.forward(output, target)

    acc_value = calculate_acc(output, labels, np.sum(test_mask))

    msg = '    Testing, total mean loss %.5f, total acc %.5f' % (
        loss_value, acc_value)
    LOG_INFO(msg)
