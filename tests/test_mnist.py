import sys

import numpy as np

sys.path.append('./dataset')
import mnist

def test_mnist_load():
    mnist.load_data("mnist")