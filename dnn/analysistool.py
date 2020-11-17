import os

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import neural_network as nn

def draw(y):
    x = list()
    for i in range(len(y)):
        x.append(i)
    fig = plt.figure(figsize=(8.0, 6.0))
    ax1 = fig.add_subplot(111)
    plt.yscale("log")
    ax2 = ax1.twinx()
    ax1.plot(x, y, "cornflowerblue")
    ax2.plot(x, y, "hotpink")
    plt.title("loss func-ish")
    plt.savefig(os.path.join(os.path.dirname(__file__), '../out/cost.png'))
    #plt.show()

def accurancygraph(y):
    x = []
    for i in range(len(y)):
        x.append(i)
    fig = plt.figure(figsize=(8.0, 6.0))
    ax1 = fig.add_subplot(111)
    ax1.plot(x, y, "mediumseagreen")
    plt.title("accurancy")
    plt.savefig(os.path.join(os.path.dirname(__file__), '../out/accurancy.png'))
    #plt.show()

def tdchart(nn):
    if (nn.structure[-1] == 1 or nn.structure[0] != 3):
        print("tdchart : dimention is not right")
    else:
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.set_xlabel('x0')
        ax.set_ylabel('x1')
        ax.set_zlabel('y')
        dence = 30
        for h in range(len(nn.z[-1])):
            x = []
            y = []
            z = []
            for i in range(dence):
                for j in range(dence):
                    x.append(i / dence)
                    y.append(j / dence)
                    nn.forwardpropagation([i / dence, j / dence], False)
                    z.append(nn.z[-1][h])
            ax.scatter3D(x, y, z, label=h)
            ax.legend()
        plt.savefig(os.path.join(os.path.dirname(__file__), '../out/3d.png'))
        #plt.show()
