import sys
import io

import numpy as np

sys.path.append('./dataset')
sys.path.append('./tools')
sys.path.append('./shared')
import analysistool as atool
import convolution_layer as cl
import convolutional_neural_network as cnn
import cleaner
import dataset
import doc_maker
import fully_connenct_layer as fc
import normalization_layer as nl
import pooling_layer as pl
import mnist
import unbuffered

# output system set
s, timestamp, commitid, branchname = doc_maker.getdata("cnn")
stdout_stream = io.StringIO()
sys.stdout = unbuffered.Unbuffered(sys.stdout, stdout_stream)

# make data
data_name = "cnn_ex"
datasetsize = 20
batch = 4
trainData, trainLabel, testData, testLabel = dataset.image(data_name, datasetsize, batch)
# generate each layer
conv = cl.Convolution_Layer(in_channel=1, out_channel=8, ksize=3, pad=1)
pool = pl.Pooling_Layer(pooling_size=[2, 2])
norm = nl.Normalization_Layer()
fullc = fc.Fully_Connect_Layer([32 + 1, 10, 4])
mycnn = cnn.Convolutional_Neural_Network([conv, pool, norm, fullc])

# train and test
epoch = 100
for i in range(epoch):
    mycnn.train(trainData, trainLabel)
    mycnn.test(testData, testLabel)

# draw graph
doc_maker.docmaker(s, timestamp, stdout_stream.getvalue(), commitid, branchname)
atool.draw(fullc.cost, timestamp)
atool.accurancygraph(fullc.accurancy, timestamp)
atool.kernelmove(conv.move, timestamp)
cleaner.clean()
