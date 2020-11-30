import sys

import numpy as np

sys.path.append("./dataset")
sys.path.append("./shared")
import activationfunction as af
import neural_network as nn
import dataset

datasize = 100
batch = 10
logic = "or"
# set data
trainData, trainLabel = dataset.logic(logic,datasize,batch)
testData, testLabel = dataset.logictest(logic,10)

# ニューラルネットワークの生成
structure = [2 + 1, 5, 2]
myNN = nn.Neural_Network(structure)
# # 学習
epoch = 3000
for i in range(epoch):
    myNN.train(trainData, trainLabel)
    myNN.test(testData, testLabel)
