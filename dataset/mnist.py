import os

import numpy as np

# Load MNIST data
def load_data():
    path = os.path.join(os.path.dirname(__file__), 'input/mnist.npz')
    with np.load(path, allow_pickle=True) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']
    return (x_train, y_train), (x_test, y_test)
