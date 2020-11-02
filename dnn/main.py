import sys

sys.path.append("./dataset")
import activationfunction as af
import analysistool as atool
import files
import logic_circuit as lc
import neural_network_engine as nne

layer = [3, 3, 1]
epoch = 30
logic = "xor"
# set data
trainData = lc.dset(logic, epoch)
testData = lc.dset(logic, 20)
# ニューラルネットワークの生成
orNN = nne.Neural_Network(layer)
# 学習
count = 20
# ニューラルネットワークのトレーニングデータ、レイヤー、重み番号、活性化関数番号の設定,cost func
orNN.model(trainData, testData, "xivier", "sigmoid", "rss")
for i in range(count):
    orNN.train()
    orNN.test()
atool.draw(orNN.cost)
atool.tdchart(orNN)
files.save(orNN)