import sys

import numpy as np

sys.path.append('./dnn')
import analysistool as atool

def test_draw():
    y = [9, 0, 8, 1, 7, 2, 6, 3, 5, 4]
    atool.draw(y)

def test_accurancygraph():
    y = [0, 0.2, 0.6, 0.7, 0.1, 0.9, 0.7, 0.3]
    atool.accurancygraph(y)
