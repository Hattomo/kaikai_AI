import sys

import numpy as np
import pytest

sys.path.append('./dataset')
import mnist

def test_mnist_load():
    mnist.load_data()