import os

import matplotlib.pyplot as plt

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