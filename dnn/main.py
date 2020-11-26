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
epoch = 100
logic = "or"
# set data
Data, Label = lc.dset(logic, 100)
trainData,trainLabel = Data.reshape([epoch,batch,-1]),Label.reshape([epoch,batch,-1])

# # ニューラルネットワークの生成
myNN = nn.Neural_Network(structure,batch,dropout)

# # 学習
epoch = 10
for i in range(epoch):
    myNN.train(trainData, trainLabel)
    myNN.test(trainData, trainLabel)
