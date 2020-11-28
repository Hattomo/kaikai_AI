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