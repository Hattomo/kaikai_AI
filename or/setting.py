# 重みと中間層の作成
import numpy as np
import random
import math

# 重みの作成
#一様分布
def wset_unif(i_node,o_node):
    weight = np.ones((o_node,i_node))
    for i in range(o_node):
        for j in range(i_node):
            weight[i][j] = random.randint(-100,100)/100
    return weight
#正規分布(xivier)
def wset_xivier(i_node,o_node):
    weight = np.random.normal(loc=0.0,scale=1/math.sqrt(i_node),size=i_node*o_node)
    weight = weight.reshape(o_node,i_node)
    return weight
#正規分布(he)
def wset_he(i_node,o_node):
    weight = np.random.normal(loc=0.0,scale=math.sqrt(2/i_node),size=i_node*o_node)
    weight = weight.reshape(o_node,i_node)
    return weight
# 中間層の作成
def layer(node):
    ans = np.ones(node)
    return ans
