import sys

import numpy as np

sys.path.append('./shared')
import vectormath as vmath

def test_vectormath():
    test_vvmat = vmath.vvmat(
        np.array([2, 3]),
        np.array([1, 4]),
    )

    assert (test_vvmat == np.array([[2, 8], [3, 12]])).all()

if __name__ == "__main__":
    test_vectormath()