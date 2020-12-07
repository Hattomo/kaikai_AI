import math
import sys

import numpy as np

sys.path.append(". / shared ")
import numpy_files as npfiles

def select_kernel(k_method, in_channel, out_channel, ksize):
    if k_method == "xavier":
        kernel = xavier(in_channel * out_channel * ksize**2, out_channel)
    elif k_method == "he":
        kernel = he(in_channel * out_channel * ksize**2, out_channel)
    elif k_method == "unif":
        kernel = unif(in_channel * out_channel * ksize**2)
    else:
        sys.stdout.write("Error: The kernel method is not found\n")
        sys.exit(1)
    # reshape
    return kernel.reshape([out_channel, in_channel, ksize, ksize])

# set all weight
def set_weight(structure, w_method):
    # set how to make weight
    if w_method == "xavier":
        w_method = xavier
    elif w_method == "he":
        w_method = he
    elif w_method == "unif":
        w_method = unif
    elif w_method == "test":
        w_method = test
    else:
        return npfiles.load(w_method)
    # make all weight
    weight_num = len(structure)
    weight_layer = list()
    for i in range(weight_num - 2):
        w = w_method(structure[i] * (structure[i + 1] - 1), structure[i + 1] - 1)
        weight_layer.append(w.reshape([structure[i + 1] - 1, structure[i]]))
    last_w = w_method(structure[-2] * structure[-1], structure[-1])
    weight_layer.append(last_w.reshape([structure[-1], structure[-2]]))
    return weight_layer

# set all layer
def set_layer(structure, batch):
    layer_num = len(structure)
    znet, ynet = list(), list()
    for i in range(layer_num):
        x, y = np.ones([batch, structure[i]]), np.ones([batch, structure[i]])
        znet.append(x)
        ynet.append(y)
    return znet, ynet

# drop out matrix
def donet(layer):
    length = len(layer)
    net = list()
    for i in range(length - 1):
        w = np.identity(layer[i])
        net.append(w)
    return net

#一様分布
def unif(num, scalesize):
    scalesize = 2
    return np.random.rand(num) * scalsesize - scalesize/2

#正規分布(xavier)
def xavier(num, scalesize):
    return np.random.normal(loc=0.0, scale=1 / math.sqrt(scalesize), size=num)

#正規分布(he)
def he(num, scalesize):
    return np.random.normal(loc=0.0, scale=1 / math.sqrt(2 / scalesize), size=num)

def test(num, scalesize):
    weight = np.zeros(num)
    for i in range(num):
        weight[i] = (i+scalesize) / 100
    return weight
