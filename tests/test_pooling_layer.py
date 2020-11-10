import sys

import numpy as np
import pytest

sys.path.append('./cnn')
sys.path.append("./shared")
import pooling_layer as pl

def test_pooling_forwordpropagation():
    a = np.zeros([1, 4, 4])
    count = 0
    for i in range(4):
        for j in range(4):
            a[0][i][j] = count
            count += 1
    pool = pl.Pooling_Layer([2, 2])
    out = pool.forwordpropagation(a)
    ans = np.array([[[5., 7.], [13., 15.]]])
    assert ((out - ans) < 1e-3).all()
