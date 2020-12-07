import os
import sys

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sys.path.append('./dnn')
import neural_network as nn

# CNN analysis tool

def draw(y, timestamp):
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
    plt.savefig(os.path.join(os.path.dirname(__file__), f'../out/{timestamp}_cost.png'))
    #plt.show()

def accurancygraph(y, timestamp):
    x = []
    for i in range(len(y)):
        x.append(i)
    fig = plt.figure(figsize=(8.0, 6.0))
    ax1 = fig.add_subplot(111)
    ax1.plot(x, y, "mediumseagreen")
    plt.title("accurancy")
    plt.savefig(os.path.join(os.path.dirname(__file__), f'../out/{timestamp}_accurancy.png'))
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

# CNN analysis tool

def show_data_gui(digit, data, data_label):
    label = str(data_label[digit])
    plt.title(str(digit) + "th Data / " + "label : " + label)
    plt.imshow(data[digit][0], cmap=plt.cm.binary)
    plt.show()

def show_basic_info(train_data):
    print("Data ndim：" + str(train_data.ndim))
    print("Data shape：" + str(train_data.shape))
    print("Data data type：" + str(train_data.dtype))
    print("Label ndim：" + str(train_label.ndim))
    print("Label shape：" + str(train_label.shape))
    print("Label data type：" + str(train_label.dtype))

def kernelmove(y, timestamp):
    x = list()
    for i in range(len(y[0])):
        x.append(i)
    fig = plt.figure(figsize=(8.0, 6.0))
    ax = fig.add_subplot(111)
    ax.plot(x, y[0], "cornflowerblue")
    ax.plot(x, y[1], "hotpink")
    plt.title("kernel max&min")
    plt.savefig(os.path.join(os.path.dirname(__file__), f'../out/{timestamp}_kernelmax&min.png'))

    fig = plt.figure(figsize=(8.0, 6.0))
    ax = fig.add_subplot(111)
    ax.plot(x, y[2], "orange")
    plt.title("kernel move")
    plt.savefig(os.path.join(os.path.dirname(__file__), f'../out/{timestamp}_kernelmove.png'))
    #plt.show()