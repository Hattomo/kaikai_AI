import os
import sys
import pickle

# save wieght
def save(weight, timestamp):
    path = os.path.join(os.path.dirname(__file__), f'../out/{timestamp}_weight.kaiai')
    f = open(path, 'wb')
    pickle.dump(weight, f)
    f.close()

# load weight
def load(path):
    path = os.path.join(os.path.dirname(__file__), path)
    f = open(path, "rb")
    load_data = pickle.load(f)
    f.close()
    return load_data
