import sys
import os
import datetime
import io

import numpy as np

sys.path.append("./dataset")
sys.path.append("./shared")
sys.path.append("./tools")
import activationfunction as af
import analysistool as atool
import neural_network as nn
import numpy_files as npfiles
import logic_circuit as lc
import doc_maker
import cleaner

class Unbuffered:

    def __init__(self, stream):
        self.stream = stream

    def write(self, data):
        self.stream.write(data)
        self.stream.flush()
        s2.write(data)
        #f.write(data)

    def flush(self):
        pass

file_path = os.path.abspath(__file__)
timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
with open(file_path) as f:
    s = f.read()

global s2
s2 = io.StringIO()
sys.stdout = Unbuffered(sys.stdout)

structure = [16 + 1, 5, 4]
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
count = 10
for i in range(count):
    orNN.train(trainData, trainLabel)
    orNN.test(testData, testLabel)

doc_maker.docmaker(s, timestamp, s2.getvalue())
atool.draw(orNN.cost, timestamp)
atool.accurancygraph(orNN.accurancy, timestamp)
atool.tdchart(orNN)
npfiles.save(orNN)
cleaner.clean()