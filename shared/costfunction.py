import random

import numpy as np

def set_costfunc(costfunc):
    if costfunc == "rss":
        return (rss, diffrss)
    elif costfunc == "cross_entropy":
        return (cross_entropy, diffcross_entropy)
    elif costfunc == "rss_sdg":
        return (rss_sdg, diffrss_sdg)
    sys.stdout.write("Error: The lossfunc is not found\n")
    sys.exit(1)

def rss(label, ans):
    return np.sum((ans - label)**2)

def diffrss(label, ans):
    return ans - label

def cross_entropy(label, ans):
    if ans.any() < 0:
        sys.stdout.write(
            "Calculation Error(cross entropy): The actfunc is not right.\nplease change actfunc of return only plus\n")
        sys.exit(1)
    return np.sum(-label * np.log(ans) - (1-label) * np.log(1 - ans))

def diffcross_entropy(label, ans):
    return -label / (ans) + (1-label) / (1-ans)

class Cost_Adam():

    def __init__(self, batchsize, label_size):
        self.batchsize = batchsize
        self.choiced_list = random.sample(range(label_size), batchsize)

    def rss_sdg(self, label, ans):
        error = 0
        for w in self.choiced_list:
            error += (ans[w] - label[w])**2
        return error

    def diffrss_sdg(self, label, ans):
        error = [0] * 4
        for w in self.choiced_list:
            error[w] += (ans[w] - label[w])
        return error