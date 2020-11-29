import sys

import numpy as np

sys.path.append("./dataset")
sys.path.append('./dnn')
sys.path.append("./shared")
import activationfunction as af
import analysistool as atool
import neural_network as nn
import numpy_files as npfiles
import logic_circuit as lc

def test_main_or_with_dropout():
    structure = [3, 3, 2]
    dropout = [0, 0.5, 0]
    epoch = 5
    logic = "or"
    # set data
    (trainData, trainLabel) = lc.dset(logic, epoch)
    (testData, testLabel) = lc.dset(logic, 5)

    # randomize
    lc.data_shuffle(trainData, trainLabel)
    lc.data_shuffle(testData, testLabel)

    # ニューラルネットワークの生成
    orNN = nn.Neural_Network(structure, dropout, "he", "tanh")
    # 学習
    count = 5
    for i in range(count):
        orNN.train(trainData, trainLabel)
        orNN.test(testData, testLabel)
    # test chart
    atool.draw(orNN.cost)
    atool.tdchart(orNN)
    # test npfiles save
    npfiles.save(orNN)

def test_main_or_without_dropout():
    structure = [3, 3, 2]
    dropout = [0, 0, 0]
    epoch = 5
    logic = "or"
    # set data
    (trainData, trainLabel) = lc.dset(logic, epoch)
    (testData, testLabel) = lc.dset(logic, 5)

    # randomize
    lc.data_shuffle(trainData, trainLabel)
    lc.data_shuffle(testData, testLabel)

    # ニューラルネットワークの生成
    orNN = nn.Neural_Network(structure, dropout, "he", "tanh")
    # 学習
    count = 5
    for i in range(count):
        orNN.train(trainData, trainLabel)
        orNN.test(testData, testLabel)
    # test chart
    atool.draw(orNN.cost)
    atool.tdchart(orNN)
    # test npfiles save
    npfiles.save(orNN)