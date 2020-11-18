import os
import sys

import numpy as np

# Load MNIST data
def load_data(data_name="mnist"):
    if data_name == "mnist":
        path = os.path.join(os.path.dirname(__file__), 'input/mnist.npz')
    elif data_name == "mnist16_mean":
        path = os.path.join(os.path.dirname(__file__), 'input/mnist16_mean.npz')
    elif data_name == "mnist16_direct":
        path = os.path.join(os.path.dirname(__file__), 'input/mnist16_direct.npz')
    with np.load(path, allow_pickle=True) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']
    return (x_train, y_train), (x_test, y_test)
