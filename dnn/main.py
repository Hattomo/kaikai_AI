import sys

import numpy as np

sys.path.append("./dataset")
sys.path.append("./shared")
import activationfunction as af
import analysistool as atool
import neural_network as nn
import numpy_files as npfiles
import logic_circuit as lc

structure = [16+1, 5, 4]
dropout = [0, 0, 0]
epoch = 30
logic = "dnn_ex"
# set data
trainData, trainLabel = lc.dset(logic, epoch)
testData, testLabel = lc.dset(logic, 10)

# randomize
lc.data_shuffle(trainData, trainLabel)
lc.data_shuffle(testData, testLabel)

# ニューラルネットワークの生成
orNN = nn.Neural_Network(structure, dropout)
# 学習
count = 100
for i in range(count):
    orNN.train(trainData, trainLabel)
    orNN.test(testData, testLabel)
atool.draw(orNN.cost)
atool.accurancygraph(orNN.accurancy)
atool.tdchart(orNN)
npfiles.save(orNN)