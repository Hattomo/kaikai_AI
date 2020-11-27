from time import sleep
import sys
import os

import numpy as np
import matplotlib.pyplot as plt

sys.path.append('./cnn')
sys.path.append('./dataset')
import cnn_analysistool as ctool
import convolution_layer as cl
import logic_circuit as lc

def zero_padding(pad, train_data):
    (datasize, channel, height, width) = np.shape(train_data)
    # <zero padding>
    p_result = np.zeros([datasize, channel, height + 2*pad, width + 2*pad])
    for i in range(datasize):
        p_result[i][0][pad:height + pad, pad:width + pad] = train_data[i]
    return p_result

def convert2mnist(data):
    if mode == "mnist16_mean" or mode == "mnist16_direct":
        newdata = np.zeros([data.shape[0], data.shape[1], 16, 16])
        for i in range(data.shape[0]):
            for j in range(0, 16):
                for k in range(0, 16):
                    if (mode == "mnist16_mean"):
                        newdata[i][0][j][k] = np.mean(data[i, 0, 2 * j:2*j + 2, 2 * k:2*k + 2])
                    elif (mode == "mnist16_direct"):
                        newdata[i][0][j][k] = data[i][0][2 * j][2 * k]
            if (i % 100 == 0):
                sys.stdout.write('\r')
                # the exact output you're looking for:
                sys.stdout.write("[%-40s] %d%%" % ('*' * int(i / (data.shape[0] - 1) * 40) + "🎄",
                                                   (i / data.shape[0] * 100) + 1))
                sys.stdout.flush()
    if mode == "mnist8_mean" or mode == "mnist8_direct":
        newdata = np.zeros([data.shape[0], data.shape[1], 8, 8])
        for i in range(data.shape[0]):
            for j in range(0, 8):
                for k in range(0, 8):
                    if (mode == "mnist8_mean"):
                        newdata[i][0][j][k] = np.mean(data[i, 0, 4 * j:4*j + 4, 4 * k:4*k + 4])
                    elif (mode == "mnist8_direct"):
                        newdata[i][0][j][k] = data[i][0][4 * j][4 * k]
            if (i % 100 == 0):
                sys.stdout.write('\r')
                # the exact output you're looking for:
                sys.stdout.write("[%-40s] %d%%" % ('*' * int(i / (data.shape[0] - 1) * 40) + "🎄",
                                                   (i / data.shape[0] * 100) + 1))
                sys.stdout.flush()
    print("")
    return newdata

def save():
    np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
    if (mode == "mnist16_mean"):
        path = os.path.join(os.path.dirname(__file__), '../dataset/input/mnist16_mean')
    elif (mode == "mnist16_direct"):
        path = os.path.join(os.path.dirname(__file__), '../dataset/input/mnist16_direct')
    if (mode == "mnist8_mean"):
        path = os.path.join(os.path.dirname(__file__), '../dataset/input/mnist8_mean')
    elif (mode == "mnist8_direct"):
        path = os.path.join(os.path.dirname(__file__), '../dataset/input/mnist8_direct')
    elif (mode == "mnist"):
        path = os.path.join(os.path.dirname(__file__), '../dataset/input/mnist28')
    np.savez(
        path,
        x_train=trainData,
        y_train=trainLabel,
        x_test=trainData,
        y_test=trainLabel,
    )
    np.warnings.filterwarnings('default', category=np.VisibleDeprecationWarning)

method = ["mnist", "mnist16_mean", "mnist16_direct", "mnist8_mean", "mnist8_direct"]
args = sys.argv
if len(args) != 2 or args[1] not in method:
    sys.stdout.write(f"Error : useage  \"python3 tools/image_resize.py {method}\"\n")
    sys.exit(5)

print("\nLoading data ...", end="")
(trainData, trainLabel) = lc.dset("mnist_train", 60000)
(testData, testLabel) = lc.dset("mnist_test", 10000)
print("  Done😀")
trainData = trainData[:, np.newaxis]
testData = testData[:, np.newaxis]

mode = args[1]
if not mode == "mnist":
    trainData = zero_padding(2, trainData)
    testData = zero_padding(2, testData)
    print("\nConverting Train data...")
    trainData = convert2mnist(trainData)
    print("\nConverting Tests data...")
    testData = convert2mnist(testData)

save()
print("\nSuccess 🎉")