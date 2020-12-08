import os
import sys

import numpy as np

# Load MNIST data
def load_data(data_name):
    if data_name == "mnist":
        path = os.path.join(os.path.dirname(__file__), 'input/mnist.npz')
    elif data_name == "mnist28":
        path = os.path.join(os.path.dirname(__file__), 'input/mnist28.npz')
    elif data_name == "mnist16_mean":
        path = os.path.join(os.path.dirname(__file__), 'input/mnist16_mean.npz')
    elif data_name == "mnist16_direct":
        path = os.path.join(os.path.dirname(__file__), 'input/mnist16_direct.npz')
    elif data_name == "mnist8_mean":
        path = os.path.join(os.path.dirname(__file__), 'input/mnist8_mean.npz')
    elif data_name == "mnist8_direct":
        path = os.path.join(os.path.dirname(__file__), 'input/mnist8_direct.npz')
    with np.load(path, allow_pickle=True) as f:
        train_data, train_label = f['x_train'], f['y_train']
        test_data, test_label = f['x_test'], f['y_test']
    return (train_data, train_label), (test_data, test_label)
