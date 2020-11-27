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
batch = 4
epoch = 300
logic = "and"
# set data
Data, Label = lc.dset(logic, epoch * batch)
D_num, D_length = Data.shape
L_num, L_length = Label.shape
trainData = np.zeros([epoch, batch, D_length])
trainLabel = np.zeros([epoch, batch, L_length])
count = 0
for i in range(epoch):
    for j in range(batch):
        trainData[i][j] = Data[count]
        trainLabel[i][j] = Label[count]
        count += 1
testData, testLabel = lc.dset(logic, batch // 4)
# # ニューラルネットワークの生成
myNN = nn.Neural_Network(structure, batch, dropout, w_method="he", actfunc="tanh")
# # 学習

for i in range(epoch):
    myNN.train(trainData[i], trainLabel[i])
    myNN.test(testData, testLabel)
atool.draw(orNN.cost)
atool.accurancygraph(orNN.accurancy)
atool.tdchart(orNN)