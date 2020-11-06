import sys

sys.path.append("./dataset")
import activationfunction as af
import analysistool as atool
import neural_network as nn
import files
import logic_circuit as lc

structure = [3, 3, 1]
epoch = 30
logic = "or"
# set data
(trainData, trainLabel) = lc.dset(logic, epoch)
(testData, testLabel) = lc.dset(logic, 20)
# ニューラルネットワークの生成
orNN = nn.Neural_Network(structure)
# 学習
count = 200
# ニューラルネットワークのトレーニングデータ、レイヤー、重み番号、活性化関数番号の設定,cost func
orNN.model("xivier", "sigmoid", "rss")
for i in range(count):
    orNN.train(trainData, trainLabel)
    orNN.test(testData, testLabel)
atool.draw(orNN.cost)
atool.tdchart(orNN)
files.save(orNN)