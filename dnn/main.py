import sys
import os
import datetime
import time

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

file_path = os.path.abspath(__file__)
timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
with open(file_path) as f:
    s = f.read()
    print(type(s))
    print(s)

te = open('log.txt', 'w')  # File where you need to keep the logs

class Unbuffered:

    def __init__(self, stream):

        self.stream = stream

    def write(self, data):

        self.stream.write(data)
        self.stream.flush()
        te.write(data)  # Write the data of stdout here to a text file as well

    def flush(self):
        pass

sys.stdout = Unbuffered(sys.stdout)
structure = [3, 3, 2]
dropout = [0, 0, 0]
epoch = 30
logic = "or"
# set data
trainData, trainLabel = lc.dset(logic, epoch)
testData, testLabel = lc.dset(logic, 10)

# randomize
lc.data_shuffle(trainData, trainLabel)
lc.data_shuffle(testData, testLabel)

# ニューラルネットワークの生成
orNN = nn.Neural_Network(structure, dropout, "he", "tanh")
# 学習
count = 30
for i in range(count):
    orNN.train(trainData, trainLabel)
    orNN.test(testData, testLabel)
    # time.sleep(5)
doc_maker.docmaker(s, timestamp)
atool.draw(orNN.cost, timestamp)
atool.accurancygraph(orNN.accurancy, timestamp)
atool.tdchart(orNN)
npfiles.save(orNN)
cleaner.clean()