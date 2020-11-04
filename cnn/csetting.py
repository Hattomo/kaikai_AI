# 重みと中間層の作成
import math
import random

import numpy as np

# 重みの作成
#一様分布
def unif(i_node, o_node):
    weight = np.random.rand(i_node * o_node).astype('f16') * 2 - 1
    weight = weight.reshape(o_node, i_node).astype('f16')
    return weight

#正規分布(xivier)
def xivier(i_node, o_node):
    weight = np.random.normal(loc=0.0, scale=1 / math.sqrt(i_node), size=i_node * o_node).astype('f16')
    weight = weight.reshape(o_node, i_node).astype('f16')
    return weight

#正規分布(he)
def he(i_node, o_node):
    weight = np.random.normal(loc=0.0, scale=math.sqrt(2 / i_node), size=i_node * o_node).astype('f16')
    weight = weight.reshape(o_node, i_node).astype('f16')
    return weight

# y
def ynet(layer):
    length = len(layer)
    net = list("-")
    for i in range(1, length):
        y = np.ones(layer[i], dtype=np.float128)
        net.append(y)
    return net

# x, z
def znet(layer):
    net = list()
    for i in range(len(layer)):
        x = np.zeros(layer[i], dtype=np.float128)
        net.append(x)
    return net

# すべての重み
def wnet(layer, w_method):
    length = len(layer)
    net = list()
    for i in range(length - 2):
        w = w_method(layer[i], layer[i + 1] - 1)
        net.append(w)
    last_w = w_method(layer[-2], layer[-1])
    net.append(last_w)
    return net
