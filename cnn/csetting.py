# 重みと中間層の作成
import math
import random

import numpy as np

# 重みの作成
#一様分布
def unif(kernel_w_size, kernel_h_size):
    kernel = np.random.rand(kernel_w_size * kernel_h_size) * 2 - 1
    kernel = kernel.reshape(kernel_w_size, kernel_h_size)
    return kernel

#正規分布(xivier)
def xivier(kernel_w_size, kernel_h_size):
    kernel = np.random.normal(loc=0.0, scale=1 / math.sqrt(kernel_w_size), size=kernel_w_size * kernel_h_size)
    kernel = kernel.reshape(kernel_w_size, kernel_h_size)
    return kernel

#正規分布(he)
def he(kernel_w_size, kernel_h_size):
    kernel = np.random.normal(loc=0.0, scale=math.sqrt(2 / kernel_w_size), size=kernel_w_size * kernel_h_size)
    kernel = kernel.reshape(kernel_w_size, kernel_h_size)
    return kernel

# x
def xnet(layer):
    length = len(layer)
    net = list()
    for i in range(1, length):
        y = np.ones(layer[i])
        net.append(y)
    return net

# y
def ynet(layer):
    net = list()
    for i in range(len(layer)):
        x = np.zeros(layer[i])
        net.append(x)
    return net

# すべての重み
def knet(channel, kernel_size, k_method):
    net = list()
    for i in range(channel):
        k = k_method(kernel_size, kernel_size)
        net.append(k)
    return net
