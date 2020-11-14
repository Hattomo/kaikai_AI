import sys

import numpy as np

sys.path.append('./shared')
import costfunction

def test_rss():
    pass

def test_cross_entropy():
    label = np.array([0, 0, 1, 1])
    ans = np.array([0.2, 0.3, 0.4, 0.5])
    error = costfunction.cross_entropy(label, ans)
    print(error)
    assert np.isclose(error, 2.189, rtol=1e-3, atol=1e-3), "check cross_entropy or test"

def test_diffcross_entropy():
    label = np.array([0, 0, 1, 1])
    ans = np.array([0.2, 0.3, 0.4, 0.5])
    error = costfunction.diffcross_entropy(label, ans)
    assert np.isclose(error, np.array([1.250, 1.428, -2.500, -2.000]), rtol=1e-3,
                      atol=1e-3).all(), "check diffcross_entropy or test"

test_cross_entropy()
test_diffcross_entropy()