# 活性化関数一覧
import numpy as np
import math
## 活性化関数

# sigmoid関数
def sigmoid(x):
    return 1/(1+math.e**-x)
# msigmoid(一次元配列 x)
def msigmoid(x):
    ans = x
    node = len(x)
    for i in range(1,node):
        ans[i] = sigmoid(x[i])
        return ans

## 活性化関数の微分

# sigmoid関数の微分
def diffsigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))
# diffsigmoid(一次元配列 x)
def mdiffsigmoid(x):
    ans = x
    node = len(x)
    for i in range(node):
        ans[i] = diffsigmoid(x[i])
    return ans
