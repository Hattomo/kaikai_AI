import os
import sys
import pickle

sys.path.append('./shared')
import numpy_files as npfiles
import neural_network as nn

def test_save_weight():
    structure = [2 + 1, 3, 3]
    test_nn = nn.Neural_Network(structure)
    npfiles.save(test_nn.weight, "test")

def test_load_weight():
    npfiles.load('../out/test_weight.kaiai')
