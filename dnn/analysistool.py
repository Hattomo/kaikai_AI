import os

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import neural_network as nn

def draw(y):
    x = list()
    for i in range(len(y)):
        x.append(i)
    plt.plot(x, y)
    plt.savefig(os.path.join(os.path.dirname(__file__), '../out/cost.png'))
    #plt.show()

def tdchart(nn):
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_xlabel('x0')
    ax.set_ylabel('x1')
    ax.set_zlabel('y')
    x = []
    y = []
    z = []
    dence = 30
    for i in range(dence):
        for j in range(dence):
            x.append(i / dence)
            y.append(j / dence)
            nn.forwordpropagation([i / dence, j / dence])
            z.append(nn.z[-1])
    ax.scatter3D(x, y, z)
    plt.savefig(os.path.join(os.path.dirname(__file__), '../out/3d.png'))
    #plt.show()
