import math
import numpy as np

# Residual Sum-Of-Squares
def rss(label, ans):
    sum_ = 0
    if len(label) == 1:
        return (ans - label)**2
    for i in range(len(label)):
        sum_ += (ans[i] - label[i])**2
    return sum_

# diff of Residual Sum-Of-Squares
def diffrss(label, ans):
    return ans - label

def non_universal_cross_entropy(label, ans):
    if ans < 0:
        sys.stdout.write(
            "Calculation Error(cross entropy): The actfunc is not right.\nplease change actfunc of return only plus\n")
        sys.exit(1)
    return -label * math.log(ans) - (1-label) * math.log(1 - ans)

def cross_entropy(label, ans):
    centropy = np.vectorize(non_universal_cross_entropy)
    return np.sum(centropy(label, ans))

def non_universal_diffcross_entropy(label, ans):
    return -label / (ans) + (1-label) / (1-ans)

def diffcross_entropy(label, ans):
    diffcentropy = np.vectorize(non_universal_diffcross_entropy)
    return diffcentropy(label, ans)