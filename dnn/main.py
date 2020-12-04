import sys
import io

import numpy as np

sys.path.append("./dataset")
sys.path.append("./shared")
sys.path.append("./tools")
import activationfunction as af
import analysistool as atool
import neural_network as nn
import dataset
import numpy_files as npfiles
import logic_circuit as lc
import doc_maker
import unbuffered
import cleaner

s, timestamp, commitid, branchname = doc_maker.getdata("dnn")
stdout_stream = io.StringIO()
sys.stdout = unbuffered.Unbuffered(sys.stdout, stdout_stream)

datasize = 4
batch = 4
logic = "and"
# set data
trainData, trainLabel = dataset.logic(logic, datasize, batch)
testData, testLabel = dataset.logictest(logic, 10)
# ニューラルネットワークの生成
structure = [2 + 1, 5, 2]
myNN = nn.Neural_Network(structure)
# # 学習
epoch = 1
for i in range(epoch):
    myNN.train(trainData, trainLabel)
    myNN.test(testData, testLabel)

doc_maker.docmaker(s, timestamp, stdout_stream.getvalue(), commitid, branchname)
atool.draw(orNN.cost, timestamp)
atool.accurancygraph(orNN.accurancy, timestamp)
atool.tdchart(orNN)
npfiles.save(orNN)
cleaner.clean()
