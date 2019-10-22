from __future__ import division
from __future__ import print_function
import numpy as np
from datetime import datetime


def onehot_encoding(label, max_num_class):
    encoding = np.eye(max_num_class)
    encoding = encoding[label]
    return encoding


def calculate_acc(output, label, num_masked_instances):
    correct = np.sum(np.argmax(output, axis=1) == label) - (len(label) - num_masked_instances)
    return correct / num_masked_instances


def LOG_INFO(msg):
    now = datetime.now()
    display_now = str(now).split(' ')[1][:-7]
    print(display_now + ' ' + msg)