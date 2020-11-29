import numpy as np

def set_costfunc(costfunc):
    if costfunc == "rss":
        return (rss, diffrss)
    elif costfunc == "cross_entropy":
        return (cross_entropy, diffcross_entropy)
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