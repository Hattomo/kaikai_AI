import math

import numpy as np

def vvmat(x, y):
    ans = np.zeros([len(x), len(y)])
    for i in range(len(x)):
        for j in range(len(y)):
            ans[i][j] = x[i] * y[j]
    return ans