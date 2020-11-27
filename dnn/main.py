import sys

import numpy as np

sys.path.append("./dataset")
sys.path.append("./shared")
import activationfunction as af
import analysistool as atool
import neural_network as nn
import numpy_files as npfiles
import logic_circuit as lc

structure = [2 + 1, 3, 2]
dropout = [0, 0, 0]
batch = 1
epoch = 2000
logic = "and"
# set data
Data, Label = lc.dset(logic, batch * epoch // 4)
trainData, trainLabel = Data.reshape([epoch, batch, -1]), Label.reshape([epoch, batch, -1])
testData, testLabel = lc.dset(logic, 1)
# # ニューラルネットワークの生成
myNN = nn.Neural_Network(structure, batch, dropout)
# print(myNN.weight)
# # 学習
for i in range(epoch):
    myNN.train(trainData[i], trainLabel[i])
    # myNN.test(testData, testLabel)
