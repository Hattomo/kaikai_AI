# 重みと中間層の作成
import numpy as np
import random

# 重みの作成
#一様分布
def wset_unif(i_node,o_node):
    weight = np.ones((o_node,i_node))
    for i in range(o_node):
        for j in range(i_node):
            weight[i][j] = random.randint(-100,100)/100
    return weight
#正規分布

# 中間層の作成
def layer(node):
    ans = np.ones(node)
    return ans
