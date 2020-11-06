# 活性化関数一覧
import math

import numpy as np

## 活性化関数
# sigmoid関数
def non_universal_sigmoid(x):
    # avoid overflow
    sigmoid_range = 34.538776394910684
    if x <= -sigmoid_range:
        return 1e-15
    elif x >= sigmoid_range:
        return 1.0 - 1e-15
    return 1.0 / (1.0 + np.exp(-x))

# vectorize non universal sigmoid
def sigmoid(x):
    sig = np.vectorize(non_universal_sigmoid)
    return sig(x)

# msigmoid(一次元配列 x)
def msigmoid(x):
    ans = x
    node = len(x)
    if node == 1:
        ans[0] = sigmoid(x[0])
        return ans
    for i in range(1, node):
        ans[i] = sigmoid(x[i])
        return ans

# tanh関数
def tanh(x):
    e = np.e
    return (e**x - e**-x) / (e**x + e**-x)

# mtanh(一次元配列　x)
def mtanh(x):
    ans = x
    node = len(x)
    for i in range(1, node):
        ans[i] = tanh(x[i])
        return ans

# relu関数
def relu(x):
    if x < 0:
        return 0
    elif x > 1e+5:
        return 1e+9
    else:
        return x

# mrelu(一次元配列　x)
def vrelu(x):
    ans = x
    node = len(x)
    if node == 1:
        ans[0] = relu(x[0])
        return ans
    for i in range(1, node):
        ans[i] = relu(x[i])
    return ans

def mrelu(x):
    ans = x
    node = np.shape(x)
    for i in range(node[0]):
        for j in range(node[1]):
            ans[i][j] = relu(x[i][j])
    return ans

## 活性化関数の微分
# sigmoid関数の微分
def diffsigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

# mdiffsigmoid(一次元配列 x)
def mdiffsigmoid(x):
    ans = x
    node = len(x)
    for i in range(node):
        ans[i] = diffsigmoid(x[i])
    return ans

# tanh関数の微分
def difftanh(x):
    e = np.e
    return 4 / (e**x + e**-x)**2

# mdifftanh(一次元配列 x)
def mdifftanh(x):
    ans = x
    node = len(x)
    for i in range(node):
        ans[i] = difftanh(x[i])
    return ans

# relu関数の微分
def diffrelu(x):
    if x < 0:
        return 0
    else:
        return 1

# mdiffrelu(一次元配列 x)
def mdiffrelu(x):
    ans = x
    node = len(x)
    for i in range(node):
        ans[i] = diffrelu(x[i])
    return ans
