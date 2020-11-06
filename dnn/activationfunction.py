# 活性化関数一覧
import math

import numpy as np

e = np.e

## 活性化関数
# sigmoid関数
def non_universal_sigmoid(x):
    sigmoid_range = 34.538776394910684
    if x <= -sigmoid_range:
        return 1e-15
    elif x >= sigmoid_range:
        return 1.0 - 1e-15
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid(x):
    sig = np.vectorize(non_universal_sigmoid)
    return sig(x)

# tanh関数
def tanh(x):
    return (e**x - e**-x) / (e**x + e**-x)

#identity関数
def non_universal_identity(x):
    identity_range = 1e+5
    if x < -identity_range:
        return -1e+5
    elif x > identity_range:
        return 1e+5
    return x

def identity(x):
    iden = np.vectorize(non_universal_identity)
    return iden(x)

def non_universal_swish(x):
    swish_range = 1e+5
    if x < -identity_range:
        return -1e+5
    elif x < identity_range:
        return 1e+5
    return x / (1 + e ^ -x)

def swish(x):
    swi = np.vectorize(non_universal_swish)
    return swi(x)

# relu関数
def non_universal_relu(x):
    if x < 0:
        return 0
    elif x > 1e+5:
        return 1e+5
    return x

def relu(x):
    re = np.vectorize(non_universal_relu)
    return re(x)

#ELU関数
def non_universal_elu(x):
    if x >= 0 and x < 1e+5:
        return x
    elif x > 1e+5:
        return 1e+5
    elif x < 0 and x > -15:
        return e ^ x - 1
    else:
        return -1

def elu(x):
    el = np.vectorize(non_universal_elu)
    return el(x)

## 活性化関数の微分
# sigmoid関数の微分
def diffsigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

# tanh関数の微分
def difftanh(x):
    return 4 / (e**x + e**-x)**2

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

# identityの微分
def diffidentity(x):
    return 1

def diffelu(x):
    if x >= 0:
        return 1
    elif x < 0 and x > -15:
        return e ^ x
    else:
        return 0

def mdiffelu(x):
    ans = x
    node = len(x)
    for i in range(node):
        ans[i] = diffrelu(x[i])
    return ans