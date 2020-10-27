# 重みと中間層の作成
import numpy as np
import random
import math

# 重みの作成
#一様分布
def unif(i_node,o_node):
    weight = np.random.rand(i_node*o_node)*2-1
    return weight
#正規分布(xivier)
def xivier(i_node,o_node):
    weight = np.random.normal(loc=0.0,scale=1/math.sqrt(i_node),size=i_node*o_node)
    weight = weight.reshape(o_node,i_node)
    return weight
#正規分布(he)
def he(i_node,o_node):
    weight = np.random.normal(loc=0.0,scale=math.sqrt(2/i_node),size=i_node*o_node)
    weight = weight.reshape(o_node,i_node)
    return weight

# 全ての層が入っている
def lnet(layer):
    length = len(layer)
    net = list()
    for i in range(length):
        if i==0 or i==length-1:
            x = np.zeros(layer[i])
            net.append(x)
        else:
            y = np.zeros(layer[i])
            h = np.zeros(layer[i])
            net.append(y)
            net.append(h)
    return net

# すべての重み
def wnet(layer,w_method):
    length = len(layer)
    net = list()
    for i in range(length-1):
        w = w_method(layer[i],layer[i+1])
        net.append(w)
    return net
