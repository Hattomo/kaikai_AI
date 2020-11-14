import sys

import numpy as np

sys.path.append('./shared')
import costfunction

def setUp():
    # init
    pass

def tearDown():
    # dispose
    pass

def test_rss():
    pass

def test_cross_entropy():
    label = np.array([0, 0, 1, 1])
    ans = np.array([0.2, 0.3, 0.4, 0.5])
    error = costfunction.cross_entropy(label, ans)
    assert error - 2.189 < 1e-3, "check cross_entropy or test"
    error = costfunction.diffcross_entropy(label, ans)
    assert (error - [1.25, 1.428, -2.50, -2.00]).all() < 1e-3, "check diffcross_entropy or test"
