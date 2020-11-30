import sys

import numpy as np

sys.path.append("./dataset")
sys.path.append('./dnn')
sys.path.append("./shared")
import dsetting

def test_donet():
    structure = [3, 4, 2]
    donet = dsetting.donet(structure)
    ans0 = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    ans1 = np.array([[1., 0., 0., 0], [0., 1., 0., 0], [0., 0., 1., 0], [0., 0., 0., 1]])
    assert np.all(donet[0] == ans0)
    assert np.all(donet[1] == ans1)