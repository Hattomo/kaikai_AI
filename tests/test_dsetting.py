import sys

import numpy as np
import pytest

sys.path.append("./dataset")
sys.path.append('./dnn')
sys.path.append("./shared")
import dsetting

def test_donet():
    structure = [4, 4, 2]
    donet = dsetting.donet(structure)
    ans0 = np.array([[1., 1., 1.], [1., 1., 1.], [1., 1., 1.], [1., 1., 1.]])
    ans1 = np.array([[1., 1.], [1., 1.], [1., 1.], [1., 1.]])
    assert np.all(donet[0] == ans0)
    assert np.all(donet[1] == ans1)