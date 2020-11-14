import numpy as np

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