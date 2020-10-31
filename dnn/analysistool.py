from matplotlib import pyplot as plt
import neural_network as nn

def draw(y):
    fig = plt.figure()
    x = list()
    for i in range(len(y)):
        x.append(i)
    plt.plot(x,y)
    #plt.show()
    plt.savefig("dnn/out/cost.png")

def chart():
    for i in range(10):
        for j in range(10):
            pass
