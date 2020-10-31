# 活性化関数一覧
import numpy as np
import math
## 活性化関数

# sigmoid関数
def sigmoid(x):
    return 1 / (1 + math.e ** -x)
# msigmoid(一次元配列 x)
def msigmoid(x):
    ans = x
    node = len(x)
    if node == 1:
        ans[0] = sigmoid(x[0])
        return ans 
    for i in range(1,node):
        ans[i] = sigmoid(x[i])
        return ans
# tanh関数
def tanh(x):
    e = math.e
    return (e ** x - e ** -x) / (e ** x + e ** -x)
# mtanh(一次元配列　x)
def mtanh(x):
    ans = x
    node = len(x)
    for i in range(1,node):
        ans[i] = tanh(x[i])
        return ans
# ReLU関数
def ReLU(x):
    if x<0:
        return 0
    else:
        return x
# mReLU(一次元配列　x)
def mReLU(x):
    ans = x
    node = len(x)
    for i in range(1,node):
        ans[i] = ReLU(x[i])
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
    return 4 / (e ** x + e ** -x)** 2
# mdifftanh(一次元配列 x)
def mdifftanh(x):
    ans = x
    node = len(x)
    for i in range(node):
        ans[i] = difftanh(x[i])
    return ans
# ReLU関数の微分
def diffReLU(x):
    if x < 0:
        return 0
    else:
        return 1
# mdiffReLU(一次元配列 x)
def mdiffReLU(x):
    ans = x
    node = len(x)
    for i in range(node):
        ans[i] = diffReLU(x[i])
    return ans