import sys

import numpy as np

sys.path.append('./shared')
import analysistool as atool

def test_draw():
    y = [9, 0, 8, 1, 7, 2, 6, 3, 5, 4]
    atool.draw(y, "test")

def test_accurancygraph():
    y = [0, 0.2, 0.6, 0.7, 0.1, 0.9, 0.7, 0.3]
    atool.accurancygraph(y, "test")

def test_kernel_move():
    y = [[9, 0, 8, 1, 7, 2, 6, 3, 5, 4], [9, 0, 8, 1, 7, 2, 6, 3, 5, 4], [9, 0, 8, 1, 7, 2, 6, 3, 5, 4]]
    atool.kernelmove(y, "test")