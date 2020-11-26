# 重みと中間層の作成
import math
import random

import numpy as np

# set all weight
def set_weight(structure, w_method):
    # set how to make weight
    if w_method == "xivier":
        w_method = dnn_xivier
    elif w_method == "he":
        w_method = dnn_he
    elif w_method == "unif":
        w_method = dnn_unif
    else:
        return npfiles.load(w_method)
    # make all weight
    weight_num = len(structure)
    weight_layer = list()
    for i in range(weight_num - 2):
        w = w_method(structure[i], structure[i + 1] - 1)
        weight_layer.append(w)
    last_w = w_method(structure[-2], structure[-1])
    weight_layer.append(last_w)
    return weight_layer

# set all layer
def set_layer(structure,batch):
    layer_num = len(structure)
    znet, ynet = list(), list()
    for i in range(layer_num):
        x, y = np.ones([batch,structure[i]]), np.zeros([batch,structure[i]])
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

# 重みの作成
#一様分布
def dnn_unif(i_node, o_node):
    weight = np.random.rand(i_node * o_node) * 2 - 1
    weight = weight.reshape(o_node, i_node)
    return weight

#正規分布(xivier)
def dnn_xivier(i_node, o_node):
    weight = np.random.normal(loc=0.0, scale=1 / math.sqrt(i_node), size=i_node * o_node)
    weight = weight.reshape(o_node, i_node)
    return weight

#正規分布(he)
def dnn_he(i_node, o_node):
    weight = np.random.normal(loc=0.0, scale=math.sqrt(2 / i_node), size=i_node * o_node)
    weight = weight.reshape(o_node, i_node)
    return weight
