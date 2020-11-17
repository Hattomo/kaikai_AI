import sys

sys.path.append("./dataset")
sys.path.append("./shared")
import activationfunction as af
import analysistool as atool
import neural_network as nn
import numpy_files as npfiles
import logic_circuit as lc

structure = [3, 3, 2]
dropout = [0, 0, 0]
epoch = 1
logic = "or"
# set data
trainData, trainLabel = lc.dset(logic, epoch)
testData, testLabel = lc.dset(logic, 10)
# ニューラルネットワークの生成
orNN = nn.Neural_Network(structure, dropout, "he", "sigmoid")
# 学習
count = 30
for i in range(count):
    orNN.train(trainData, trainLabel)
    orNN.test(testData, testLabel)
atool.draw(orNN.cost)
atool.accurancygraph(orNN.accurancy)
atool.tdchart(orNN)
npfiles.save(orNN)