import os
import sys
import pickle

# save wieght
def save(nn):
    path = os.path.join(os.path.dirname(__file__), '../out/weight.kaiai')
    f = open(path, 'wb')
    pickle.dump(nn.weight, f)
    f.close()

# load weight
def load(path):
    path = os.path.join(os.path.dirname(__file__), path)
    f = open(path, "rb")
    load_data = pickle.load(f)
    f.close()
    return load_data
