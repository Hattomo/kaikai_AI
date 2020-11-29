import sys

import numpy as np

sys.path.append("./dataset")
sys.path.append("./shared")
import activationfunction as af
import neural_network as nn
import dataset

datasize = 4
batch = 4
logic = "original_and"
# set data
trainData, trainLabel = dataset.logic(logic,datasize,batch)
testData, testLabel = dataset.logictest(logic)
# # ニューラルネットワークの生成
structure = [2 + 1, 4, 2]
myNN = nn.Neural_Network(structure,w_method="he",actfunc="tanh")
# # 学習
epoch = 1000
for i in range(epoch):
    myNN.train(trainData, trainLabel)
    myNN.test(testData, testLabel)
