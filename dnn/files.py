import sys
import os
import numpy as np


# save wieght
def save(nn):
    np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
    path = os.path.join(os.path.dirname(__file__), 'out/np_savez')
    np.savez(path, weight=nn.weight)
    np.warnings.filterwarnings('default',
                               category=np.VisibleDeprecationWarning)


# load weight
def load(path):
    path = os.path.join(os.path.dirname(__file__), path)
    return np.load(path, allow_pickle=True)["weight"]
