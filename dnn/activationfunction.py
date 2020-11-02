# 活性化関数一覧
import math

import numpy as np

## 活性化関数
# sigmoid関数
def sigmoid(x):
    return 1 / (1 + math.e**-x)

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
    e = math.e
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
    else:
        return x

# mrelu(一次元配列　x)
def mrelu(x):
    ans = x
    node = len(x)
    if node == 1:
        ans[0] = relu(x[0])
        return ans
    for i in range(1, node):
        ans[i] = relu(x[i])
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
    e = math.e
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
