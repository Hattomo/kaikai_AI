import sys

sys.path.append("./dataset")
sys.path.append("./shared")
import activationfunction as af
import analysistool as atool
import neural_network as nn
import numpy_files as npfiles
import logic_circuit as lc

structure = [3, 3, 1]
epoch = 30
logic = "or"
# set data
(trainData, trainLabel) = lc.dset(logic, epoch)
(testData, testLabel) = lc.dset(logic, 20)
# ニューラルネットワークの生成
orNN = nn.Neural_Network(structure, "he", "tanh")
# 学習
count = 25
for i in range(count):
    orNN.train(trainData, trainLabel)
    orNN.test(testData, testLabel)
atool.draw(orNN.cost)
atool.tdchart(orNN)
npfiles.save(orNN)