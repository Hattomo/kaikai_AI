import sys
import io

import numpy as np

sys.path.append('./dnn')
sys.path.append('./dataset')
sys.path.append('./tools')
import analysistool as atool
import cnn_analysistool as catool
import convolution_layer as cl
import convolutional_neural_network as cnn
import csetting
import fully_connenct_layer as fc
import neural_network as nn
import normalization_layer as nl
import pooling_layer as pl
import mnist
import logic_circuit as lc
import doc_maker
import unbuffered
import cleaner

s, timestamp, commitid, branchname = doc_maker.getdata("cnn")
stdout_stream = io.StringIO()
sys.stdout = unbuffered.Unbuffered(sys.stdout, stdout_stream)

(trainData, trainLabel) = lc.dset("cnn_exs", 5)
(testData, testLabel) = lc.dset("cnn_exs", 1)

conv = cl.Convolution_Layer(in_channel=1, out_channel=4, ksize=3, pad=1)
pool = pl.Pooling_Layer(pooling_size=[2, 2])
fullc = fc.Fully_Connect_Layer([16 + 1, 4, 2])
normalize = nl.Normalization_Layer()

epoch = 50
for i in range(epoch):
    # train
    conv_out = conv.forwardpropagation(trainData / 255)
    pool_out = pool.forwardpropagation(conv_out)
    normalized_data = normalize.normalize(pool_out)
    error = fullc.train(normalized_data, trainLabel)
    pool_error = pool.backpropagation(error)
    conv.backpropagation(pool_error)
    # test
    conv_out = conv.forwardpropagation(testData / 255)
    pool_out = pool.forwardpropagation(conv_out)
    normalized_data = normalize.normalize(pool_out)
    fullc.test(normalized_data, trainLabel)

# draw graph
doc_maker.docmaker(s, timestamp, stdout_stream.getvalue(), commitid, branchname)
atool.draw(fullc.cost, timestamp)
atool.accurancygraph(fullc.accurancy, timestamp)
catool.kernelmove(conv.move, timestamp)
cleaner.clean()
